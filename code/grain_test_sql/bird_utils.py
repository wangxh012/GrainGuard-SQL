from __future__ import annotations

import copy
import json
import sqlite3
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from grain_test_sql.schema_registry import ForeignKey, SchemaRegistry, TableSchema
from grain_test_sql.signature_extractor import FilterSignature, extract_signature


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_lenient(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    recovered = 0
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            offset = 0
            line_rows = 0
            while offset < len(text):
                try:
                    value, end = decoder.raw_decode(text, offset)
                except json.JSONDecodeError as exc:
                    raise json.JSONDecodeError(
                        f"{exc.msg} in {path} line {line_number}",
                        exc.doc,
                        exc.pos,
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path} line {line_number} decoded a {type(value).__name__}, expected object")
                rows.append(value)
                line_rows += 1
                offset = end
                while offset < len(text) and text[offset].isspace():
                    offset += 1
            if line_rows > 1:
                recovered += line_rows - 1
    if recovered:
        print(f"Recovered {recovered} concatenated JSONL object(s) from {path}", flush=True)
    return rows


def registry_from_bird_meta(
    meta: dict[str, Any],
    sqlite_path: Path | None = None,
    *,
    profile_observed_filter_uniqueness: bool = False,
    profile_observed_fk_pairs: bool = False,
) -> SchemaRegistry:
    table_names = meta["table_names_original"]
    column_names = meta["column_names_original"]
    column_types = meta["column_types"]

    columns_by_table: dict[str, dict[str, str]] = {t: {} for t in table_names}
    global_to_ref: dict[int, tuple[str, str]] = {}
    for idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx < 0:
            continue
        table = table_names[table_idx]
        type_name = column_types[idx] if idx < len(column_types) else "text"
        columns_by_table[table][col_name] = type_name or "text"
        global_to_ref[idx] = (table, col_name)

    pk_by_table: dict[str, list[str]] = {t: [] for t in table_names}
    for pk in meta.get("primary_keys", []):
        pk_indices = pk if isinstance(pk, list) else [pk]
        for idx in pk_indices:
            if idx in global_to_ref:
                table, col = global_to_ref[idx]
                pk_by_table[table].append(col)

    unique_by_table: dict[str, list[list[str]]] = {t: [] for t in table_names}
    if sqlite_path and sqlite_path.exists():
        sqlite_constraints = _declared_sqlite_constraints(sqlite_path, columns_by_table)
        for table, constraints in sqlite_constraints.items():
            if constraints["primary_key"]:
                pk_by_table[table] = constraints["primary_key"]
            unique_by_table[table] = constraints["unique"]

    fks_by_table: dict[str, list[ForeignKey]] = {t: [] for t in table_names}
    for left_idx, right_idx in meta.get("foreign_keys", []):
        if left_idx not in global_to_ref or right_idx not in global_to_ref:
            continue
        left_table, left_col = global_to_ref[left_idx]
        right_table, right_col = global_to_ref[right_idx]
        child_table, child_col, parent_table, parent_col = orient_fk(
            left_table, left_col, right_table, right_col, pk_by_table
        )
        fks_by_table[child_table].append(
            ForeignKey(
                columns=[child_col],
                ref_table=parent_table,
                ref_columns=[parent_col],
                many_allowed=True,
            )
        )

    tables = {
        table: TableSchema(
            name=table,
            columns=columns_by_table[table],
            primary_key=pk_by_table[table],
            grain=pk_by_table[table] or infer_grain(columns_by_table[table]),
            foreign_keys=fks_by_table[table],
            unique=unique_by_table[table],
        )
        for table in table_names
    }
    registry = SchemaRegistry(tables=tables)
    if sqlite_path and sqlite_path.exists():
        calibrate_registry_uniqueness_from_sqlite(
            registry,
            sqlite_path,
            profile_all_single_columns=profile_observed_filter_uniqueness,
            profile_fk_filter_composites=profile_observed_filter_uniqueness,
            profile_fk_pairs=profile_observed_fk_pairs,
        )
    return registry


def _declared_sqlite_constraints(
    sqlite_path: Path,
    columns_by_table: dict[str, dict[str, str]],
) -> dict[str, dict[str, list[Any]]]:
    """Read complete DDL primary/unique keys, preserving metadata name casing."""
    result: dict[str, dict[str, list[Any]]] = {}
    conn = sqlite3.connect(str(sqlite_path))
    try:
        for table_name, metadata_columns in columns_by_table.items():
            if not _sqlite_table_exists(conn, table_name):
                continue
            by_lower = {column.casefold(): column for column in metadata_columns}
            table_info = conn.execute(
                f"PRAGMA table_info({_q_sql(table_name)})"
            ).fetchall()
            pk_parts = sorted(
                (int(row[5]), by_lower.get(str(row[1]).casefold(), str(row[1])))
                for row in table_info
                if row[5]
            )
            primary_key = [column for _, column in pk_parts]

            unique: list[list[str]] = []
            for index_row in conn.execute(
                f"PRAGMA index_list({_q_sql(table_name)})"
            ).fetchall():
                if not bool(index_row[2]):
                    continue
                if len(index_row) > 4 and bool(index_row[4]):
                    continue
                index_name = str(index_row[1])
                columns = [
                    by_lower.get(str(row[2]).casefold(), str(row[2]))
                    for row in conn.execute(
                        f"PRAGMA index_info({_q_sql(index_name)})"
                    ).fetchall()
                    if row[2] is not None
                ]
                if columns and columns != primary_key and columns not in unique:
                    unique.append(columns)
            result[table_name] = {
                "primary_key": primary_key,
                "unique": unique,
            }
    finally:
        conn.close()
    return result


def add_observed_fk_uniques(registry: SchemaRegistry, sqlite_path: Path) -> None:
    """Add observed unique FK columns from the current SQLite instance.

    BIRD metadata often declares primary keys and foreign keys but omits useful
    uniqueness facts such as loan.account_id being one-to-one in the released
    database. We record them separately from DDL constraints so reports can
    distinguish schema proofs from benchmark-instance profile facts.
    """
    conn = sqlite3.connect(str(sqlite_path))
    try:
        for table_name, table in registry.tables.items():
            known = {
                tuple(table.primary_key),
                *(tuple(cols) for cols in table.unique),
                *(tuple(cols) for cols in table.observed_unique),
            }
            for fk in table.foreign_keys:
                cols = tuple(fk.columns)
                if not cols or cols in known:
                    continue
                if _observed_unique_non_null(conn, table_name, list(cols)):
                    table.observed_unique.append(list(cols))
                    known.add(cols)
    finally:
        conn.close()


def calibrate_registry_uniqueness_from_sqlite(
    registry: SchemaRegistry,
    sqlite_path: Path,
    *,
    profile_all_single_columns: bool = True,
    profile_fk_filter_composites: bool = False,
    profile_fk_pairs: bool = False,
) -> None:
    """Attach observed uniqueness/non-uniqueness facts from a SQLite DB.

    The registry is schema metadata, not a proof. Some Spider/BIRD metadata
    declares single-column primary keys that are not unique in the released
    database. Those bad uniqueness facts can create false SCHEMA_PASS results.
    We preserve the declared keys for grain naming, but record observed
    non-unique columns so uniqueness checks can refuse unsafe schema passes.
    """
    conn = sqlite3.connect(str(sqlite_path))
    try:
        for table_name, table in registry.tables.items():
            if not _sqlite_table_exists(conn, table_name):
                continue
            known_non_unique = {tuple(cols) for cols in table.observed_non_unique}

            candidate_sets: list[list[str]] = []
            candidate_sets.extend(
                cols for cols in [table.primary_key, *table.unique] if cols
            )
            candidate_sets.extend(fk.columns for fk in table.foreign_keys if fk.columns)
            if profile_fk_pairs:
                fk_column_sets = [
                    fk.columns for fk in table.foreign_keys if fk.columns
                ]
                candidate_sets.extend(
                    list(dict.fromkeys([*left, *right]))
                    for index, left in enumerate(fk_column_sets)
                    for right in fk_column_sets[index + 1 :]
                )
            if profile_fk_filter_composites:
                candidate_sets.extend(
                    [*fk.columns, col]
                    for fk in table.foreign_keys
                    for col in table.columns
                    if fk.columns and col not in fk.columns
                )
            if profile_all_single_columns:
                candidate_sets.extend([col] for col in table.columns)

            seen: set[tuple[str, ...]] = set()
            for cols in candidate_sets:
                key = tuple(cols)
                if not key or key in seen:
                    continue
                seen.add(key)
                if not all(_sqlite_column_exists(conn, table_name, col) for col in cols):
                    continue
                if _observed_unique_non_null(conn, table_name, list(cols)):
                    declared = {
                        tuple(table.primary_key),
                        *(tuple(x) for x in table.unique),
                    }
                    if key not in declared and key not in {
                        tuple(x) for x in table.observed_unique
                    }:
                        table.observed_unique.append(list(cols))
                    continue
                if (
                    key not in known_non_unique
                    and _observed_has_duplicate_non_null(
                        conn,
                        table_name,
                        list(cols),
                    )
                ):
                    table.observed_non_unique.append(list(cols))
                    known_non_unique.add(key)
    finally:
        conn.close()


def calibrate_registry_filter_uniqueness_from_sql(
    registry: SchemaRegistry,
    sqlite_path: Path,
    sql: str,
) -> None:
    """Attach observed uniqueness facts needed by one SQL's equality filters.

    This is intentionally narrower than profiling an entire SQLite database. It
    only checks combinations that can make a DET event table non-duplicating for
    the current SQL, such as (uuid, language) under language = 'German', or
    (hero_id, power_id) when a joined superpower row is fixed by power_name.
    """
    if not sqlite_path.exists():
        return
    try:
        signature = extract_signature(sql)
    except Exception:
        return

    filter_columns = _filter_columns_by_table(registry, signature.filters)
    if not filter_columns:
        return

    conn = sqlite3.connect(str(sqlite_path))
    try:
        unique_filter_tables: set[str] = set()
        for table_name, columns in filter_columns.items():
            if _add_observed_unique(
                registry,
                conn,
                table_name,
                list(columns),
            ):
                unique_filter_tables.add(table_name)
            elif registry.columns_are_unique(table_name, list(columns)):
                unique_filter_tables.add(table_name)

        for table_name, table in registry.tables.items():
            direct_filter_columns = filter_columns.get(table_name, set())
            if direct_filter_columns:
                for fk in table.foreign_keys:
                    _add_observed_unique(
                        registry,
                        conn,
                        table_name,
                        [*fk.columns, *sorted(direct_filter_columns)],
                    )

            constrained_fk_columns: set[str] = set()
            for fk in table.foreign_keys:
                ref_table = registry.resolve_table_name(fk.ref_table)
                if ref_table in unique_filter_tables:
                    constrained_fk_columns.update(fk.columns)
            if constrained_fk_columns:
                for fk in table.foreign_keys:
                    _add_observed_unique(
                        registry,
                        conn,
                        table_name,
                        [*fk.columns, *sorted(constrained_fk_columns)],
                    )
    finally:
        conn.close()


def registry_with_filter_uniqueness_profile(
    registry: SchemaRegistry,
    sqlite_path: Path,
    sql: str,
    *,
    decision_eligible: bool = False,
) -> SchemaRegistry:
    """Return an isolated registry enriched only for one candidate SQL.

    The base registry remains unchanged, preventing profile facts discovered
    for one candidate from leaking into later decisions. The profile reads the
    candidate SQL and released database instance, never a reference SQL or a
    correctness label.
    """
    profiled = copy.deepcopy(registry)
    calibrate_registry_filter_uniqueness_from_sql(
        profiled,
        sqlite_path,
        sql,
    )
    profiled.observed_uniqueness_scope_enabled = True
    if decision_eligible:
        profiled.observed_uniqueness_policy = "conditional_pass"
    return profiled


def _filter_columns_by_table(
    registry: SchemaRegistry,
    filters: list[FilterSignature],
) -> dict[str, set[str]]:
    columns_by_table: dict[str, set[str]] = {}
    for filter_ in filters:
        if filter_.operator != "=" or "." not in filter_.source:
            continue
        table_name, column = filter_.source.split(".", 1)
        table_name = registry.resolve_table_name(table_name)
        if table_name not in registry.tables:
            continue
        columns_by_table.setdefault(table_name, set()).add(
            registry.resolve_column_name(table_name, column)
        )
    return columns_by_table


def _add_observed_unique(
    registry: SchemaRegistry,
    conn: sqlite3.Connection,
    table_name: str,
    columns: list[str],
) -> bool:
    table_name = registry.resolve_table_name(table_name)
    if table_name not in registry.tables:
        return False
    deduped_columns = list(dict.fromkeys(columns))
    if not deduped_columns:
        return False
    if registry.uniqueness_basis(
        table_name,
        deduped_columns,
        include_observed=True,
    ):
        return True
    if not all(_sqlite_column_exists(conn, table_name, col) for col in deduped_columns):
        return False
    if not _observed_unique_non_null(conn, table_name, deduped_columns):
        return False
    registry.tables[table_name].observed_unique.append(deduped_columns)
    return True


def _sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND lower(name) = lower(?) LIMIT 1",
            (table,),
        ).fetchone()
        is not None
    )


def _sqlite_column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info({_q_sql(table)})").fetchall()
    except sqlite3.DatabaseError:
        return False
    return any(str(row[1]).lower() == column.lower() for row in rows)


def _observed_unique_non_null(conn: sqlite3.Connection, table: str, columns: list[str]) -> bool:
    quoted_cols = [_q_sql(col) for col in columns]
    non_null_predicate = " AND ".join(f"{col} IS NOT NULL" for col in quoted_cols)
    count_sql = f"SELECT COUNT(*) FROM {_q_sql(table)} WHERE {non_null_predicate}"
    non_null_count = conn.execute(count_sql).fetchone()[0]
    if non_null_count == 0:
        return False

    grouped_cols = ", ".join(quoted_cols)
    duplicate_sql = (
        f"SELECT 1 FROM {_q_sql(table)} "
        f"WHERE {non_null_predicate} "
        f"GROUP BY {grouped_cols} HAVING COUNT(*) > 1 LIMIT 1"
    )
    return conn.execute(duplicate_sql).fetchone() is None


def _observed_has_duplicate_non_null(conn: sqlite3.Connection, table: str, columns: list[str]) -> bool:
    quoted_cols = [_q_sql(col) for col in columns]
    non_null_predicate = " AND ".join(f"{col} IS NOT NULL" for col in quoted_cols)
    grouped_cols = ", ".join(quoted_cols)
    duplicate_sql = (
        f"SELECT 1 FROM {_q_sql(table)} "
        f"WHERE {non_null_predicate} "
        f"GROUP BY {grouped_cols} HAVING COUNT(*) > 1 LIMIT 1"
    )
    return conn.execute(duplicate_sql).fetchone() is not None


def _q_sql(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def orient_fk(
    left_table: str,
    left_col: str,
    right_table: str,
    right_col: str,
    pk_by_table: dict[str, list[str]],
) -> tuple[str, str, str, str]:
    if right_col in pk_by_table.get(right_table, []):
        return left_table, left_col, right_table, right_col
    if left_col in pk_by_table.get(left_table, []):
        return right_table, right_col, left_table, left_col
    return left_table, left_col, right_table, right_col


def infer_grain(columns: dict[str, str]) -> list[str]:
    for col in columns:
        if col.lower().endswith("id"):
            return [col]
    return []


def is_analytical(sql: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return False
    has_agg = any(isinstance(x, (exp.Sum, exp.Avg, exp.Count, exp.Max, exp.Min)) for x in tree.walk())
    has_join = tree.find(exp.Join) is not None
    has_group = tree.find(exp.Group) is not None
    has_setop = any(isinstance(x, (exp.Union, exp.Intersect, exp.Except)) for x in tree.walk())
    return has_agg and (has_join or has_group) and not has_setop


def write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
