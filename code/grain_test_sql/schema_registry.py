from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ForeignKey:
    columns: list[str]
    ref_table: str
    ref_columns: list[str]
    many_allowed: bool = True


@dataclass
class TableSchema:
    name: str
    columns: dict[str, str]
    primary_key: list[str]
    grain: list[str]
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    unique: list[list[str]] = field(default_factory=list)
    observed_unique: list[list[str]] = field(default_factory=list)
    observed_non_unique: list[list[str]] = field(default_factory=list)


@dataclass
class SchemaRegistry:
    tables: dict[str, TableSchema]
    observed_uniqueness_policy: str = "provenance_only"
    observed_uniqueness_scope_enabled: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SchemaRegistry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tables: dict[str, TableSchema] = {}
        for table_name, raw in data["tables"].items():
            if table_name.lower().startswith("sqlite_"):
                continue
            fks: list[ForeignKey] = []
            for fk in raw.get("foreign_keys", []) or []:
                ref = fk["references"]
                if isinstance(ref, str):
                    ref_table, ref_col = ref.split(".", 1)
                    ref_columns = [ref_col]
                else:
                    ref_table = ref["table"]
                    ref_columns = list(ref["columns"])
                fks.append(
                    ForeignKey(
                        columns=list(fk["columns"]),
                        ref_table=ref_table,
                        ref_columns=ref_columns,
                        many_allowed=bool(fk.get("many_allowed", True)),
                    )
                )
            pk = list(raw.get("primary_key", []))
            grain = list(raw.get("grain", pk))
            tables[table_name] = TableSchema(
                name=table_name,
                columns=dict(raw.get("columns", {})),
                primary_key=pk,
                grain=grain,
                foreign_keys=fks,
                unique=[list(x) for x in raw.get("unique", []) or []],
                observed_unique=[list(x) for x in raw.get("observed_unique", []) or []],
                observed_non_unique=[list(x) for x in raw.get("observed_non_unique", []) or []],
            )
        return cls(
            tables=tables,
            observed_uniqueness_scope_enabled=bool(
                data.get("observed_uniqueness_scope_enabled", False)
            ),
        )

    def table_for_column_ref(self, ref: str) -> tuple[str, str]:
        table, col = ref.split(".", 1)
        table = self.resolve_table_name(table)
        col = self.resolve_column_name(table, col)
        if table not in self.tables or col not in self.tables[table].columns:
            raise KeyError(f"Unknown column reference: {ref}")
        return table, col

    def resolve_table_name(self, table: str) -> str:
        if table in self.tables:
            return table
        lowered = table.lower()
        for name in self.tables:
            if name.lower() == lowered:
                return name
        return table

    def resolve_column_name(self, table: str, column: str) -> str:
        table = self.resolve_table_name(table)
        if table not in self.tables:
            return column
        if column in self.tables[table].columns:
            return column
        lowered = column.lower()
        for name in self.tables[table].columns:
            if name.lower() == lowered:
                return name
        return column

    def to_dict(self) -> dict[str, Any]:
        tables: dict[str, Any] = {}
        for name, table in self.tables.items():
            tables[name] = {
                "columns": table.columns,
                "primary_key": table.primary_key,
                "grain": table.grain,
            }
            if table.unique:
                tables[name]["unique"] = table.unique
            if table.observed_unique:
                tables[name]["observed_unique"] = table.observed_unique
            if table.observed_non_unique:
                tables[name]["observed_non_unique"] = table.observed_non_unique
            if table.foreign_keys:
                tables[name]["foreign_keys"] = [
                    {
                        "columns": fk.columns,
                        "references": {"table": fk.ref_table, "columns": fk.ref_columns},
                        "many_allowed": fk.many_allowed,
                    }
                    for fk in table.foreign_keys
                ]
        result: dict[str, Any] = {"tables": tables}
        if self.observed_uniqueness_scope_enabled:
            result["observed_uniqueness_scope_enabled"] = True
        return result

    def column_exists(self, ref: str) -> bool:
        try:
            self.table_for_column_ref(ref)
            return True
        except KeyError:
            return False

    def grain_refs(self, table: str) -> list[str]:
        return [f"{table}.{col}" for col in self.tables[table].grain]

    def pk_value_column(self, table: str) -> str:
        pk = self.tables[table].primary_key
        if not pk:
            raise ValueError(f"Table {table} has no primary key")
        return pk[0]

    def children_of(self, table: str) -> list[tuple[str, ForeignKey]]:
        children: list[tuple[str, ForeignKey]] = []
        for child_name, child in self.tables.items():
            for fk in child.foreign_keys:
                if fk.ref_table == table:
                    children.append((child_name, fk))
        return children

    def columns_are_unique(self, table: str, columns: list[str]) -> bool:
        column_set = set(columns)
        schema = self.tables[table]
        contradicted = {
            frozenset(non_unique_cols)
            for non_unique_cols in schema.observed_non_unique
            if non_unique_cols
        }
        unique_sets = [schema.primary_key, *schema.unique]
        return any(
            unique
            and set(unique).issubset(column_set)
            and frozenset(unique) not in contradicted
            for unique in unique_sets
        )

    def columns_are_observed_unique(self, table: str, columns: list[str]) -> bool:
        """Return an instance-profile fact without treating it as DDL proof."""
        column_set = set(columns)
        schema = self.tables[table]
        contradicted = {
            frozenset(non_unique_cols)
            for non_unique_cols in schema.observed_non_unique
            if non_unique_cols
        }
        return any(
            unique
            and set(unique).issubset(column_set)
            and frozenset(unique) not in contradicted
            for unique in schema.observed_unique
        )

    def uniqueness_basis(
        self,
        table: str,
        columns: list[str],
        *,
        include_observed: bool = False,
    ) -> str | None:
        if self.columns_are_unique(table, columns):
            return "SCHEMA"
        if include_observed and self.columns_are_observed_unique(table, columns):
            return "BENCHMARK_PROFILE"
        return None

    def columns_are_known_non_unique(self, table: str, columns: list[str]) -> bool:
        if table not in self.tables:
            return False
        column_set = frozenset(columns)
        return any(
            non_unique_cols and frozenset(non_unique_cols) == column_set
            for non_unique_cols in self.tables[table].observed_non_unique
        )

    def entity_identifier_columns(self, table: str) -> set[str]:
        table = self.resolve_table_name(table)
        if table not in self.tables:
            return set()
        schema = self.tables[table]
        identifiers: set[str] = set(schema.primary_key)
        for unique_cols in schema.unique:
            if len(unique_cols) == 1:
                identifiers.add(unique_cols[0])
        for child in self.tables.values():
            for fk in child.foreign_keys:
                if fk.ref_table == table:
                    identifiers.update(fk.ref_columns)
        return identifiers

    def columns_identify_same_entity(self, left: str | None, right: str | None) -> bool:
        if not left or not right or "." not in right:
            return False
        right_table, right_col = right.split(".", 1)
        right_table = self.resolve_table_name(right_table)
        right_col = self.resolve_column_name(right_table, right_col)
        if right_table not in self.tables:
            return False

        if "." in left:
            left_table, left_col = left.split(".", 1)
            left_table = self.resolve_table_name(left_table)
            left_col = self.resolve_column_name(left_table, left_col)
            if left_table != right_table:
                return False
        else:
            left_col = self.resolve_column_name(right_table, left)

        if left_col == right_col:
            return True
        identifiers = self.entity_identifier_columns(right_table)
        return left_col in identifiers and right_col in identifiers

    def child_is_unique_for_parent(self, child: str, parent: str, fk: ForeignKey | None = None) -> bool:
        fk = fk or self.fk_to_parent(child, parent)
        return bool(fk and self.columns_are_unique(child, fk.columns))

    def child_is_unique_for_parent_under_filters(
        self,
        child: str,
        parent: str,
        filters: list[Any],
        fk: ForeignKey | None = None,
    ) -> bool:
        """Return whether a child table is unique per parent after equality filters.

        A table can be many-to-one in general but safe under a query filter. For
        example, foreign_data is many rows per card by uuid, but (uuid,
        language) is unique; a WHERE foreign_data.language = 'German' predicate
        makes the join non-duplicating for card counts.
        """
        return self.child_uniqueness_basis_under_filters(
            child,
            parent,
            filters,
            fk,
        ) is not None

    def child_uniqueness_basis_under_filters(
        self,
        child: str,
        parent: str,
        filters: list[Any],
        fk: ForeignKey | None = None,
    ) -> str | None:
        """Return SCHEMA or BENCHMARK_PROFILE for filtered child uniqueness."""
        child = self.resolve_table_name(child)
        parent = self.resolve_table_name(parent)
        fk = fk or self.fk_to_parent(child, parent)
        if not fk:
            return None
        include_observed = self.observed_uniqueness_policy == "conditional_pass"
        direct_basis = self.uniqueness_basis(
            child,
            fk.columns,
            include_observed=include_observed,
        )
        if direct_basis:
            return direct_basis

        constrained_columns = set(fk.columns)
        constrained_columns.update(self._equality_filter_columns(child, filters))
        profile_dependency = False

        for child_fk in self.tables[child].foreign_keys:
            ref_table = self.resolve_table_name(child_fk.ref_table)
            if ref_table == parent:
                continue
            ref_filter_columns = self._equality_filter_columns(ref_table, filters)
            ref_basis = (
                self.uniqueness_basis(
                    ref_table,
                    list(ref_filter_columns),
                    include_observed=include_observed,
                )
                if ref_filter_columns
                else None
            )
            if ref_basis:
                constrained_columns.update(child_fk.columns)
                profile_dependency = profile_dependency or ref_basis == "BENCHMARK_PROFILE"

        basis = self.uniqueness_basis(
            child,
            list(constrained_columns),
            include_observed=include_observed,
        )
        if not basis:
            return None
        if profile_dependency or basis == "BENCHMARK_PROFILE":
            return "BENCHMARK_PROFILE"
        return "SCHEMA"

    def _equality_filter_columns(self, table: str, filters: list[Any]) -> set[str]:
        table = self.resolve_table_name(table)
        columns: set[str] = set()
        for filter_ in filters:
            if getattr(filter_, "operator", None) != "=":
                continue
            source = getattr(filter_, "source", None)
            if not source or "." not in source:
                continue
            source_table, source_col = source.split(".", 1)
            source_table = self.resolve_table_name(source_table)
            if source_table != table:
                continue
            columns.add(self.resolve_column_name(table, source_col))
        return columns

    def sibling_is_unique_for_source(
        self,
        sibling: str,
        source: str,
        sibling_fk: ForeignKey | None = None,
    ) -> bool:
        return self.sibling_uniqueness_basis_for_source(
            sibling,
            source,
            sibling_fk,
        ) is not None

    def sibling_uniqueness_basis_for_source(
        self,
        sibling: str,
        source: str,
        sibling_fk: ForeignKey | None = None,
    ) -> str | None:
        sibling_fk = sibling_fk or next(
            iter(self.tables[sibling].foreign_keys),
            None,
        )
        if not sibling_fk:
            return None
        for source_fk in self.tables[source].foreign_keys:
            if (
                sibling_fk.ref_table == source_fk.ref_table
                and sibling_fk.ref_columns == source_fk.ref_columns
            ):
                return self.uniqueness_basis(sibling, sibling_fk.columns)
        return None

    def sibling_uniqueness_basis_for_source_under_filters(
        self,
        sibling: str,
        source: str,
        filters: list[Any],
        sibling_fk: ForeignKey | None = None,
    ) -> str | None:
        sibling_fk = sibling_fk or next(
            iter(self.tables[sibling].foreign_keys),
            None,
        )
        if not sibling_fk:
            return None
        for source_fk in self.tables[source].foreign_keys:
            if (
                sibling_fk.ref_table == source_fk.ref_table
                and sibling_fk.ref_columns == source_fk.ref_columns
            ):
                return self.child_uniqueness_basis_under_filters(
                    sibling,
                    sibling_fk.ref_table,
                    filters,
                    sibling_fk,
                )
        return None

    def fks_to_parent(self, child: str, parent: str) -> list[ForeignKey]:
        child = self.resolve_table_name(child)
        parent = self.resolve_table_name(parent)
        if child not in self.tables:
            return []
        return [
            fk
            for fk in self.tables[child].foreign_keys
            if self.resolve_table_name(fk.ref_table) == parent
        ]

    def fk_to_parent(
        self,
        child: str,
        parent: str,
        *,
        ref_column: str | None = None,
    ) -> ForeignKey | None:
        candidates = self.fks_to_parent(child, parent)
        if ref_column is None:
            return candidates[0] if candidates else None
        parent = self.resolve_table_name(parent)
        resolved_column = self.resolve_column_name(parent, ref_column)
        exact = next(
            (
                fk
                for fk in candidates
                if resolved_column
                in {
                    self.resolve_column_name(parent, column)
                    for column in fk.ref_columns
                }
            ),
            None,
        )
        if exact is not None:
            return exact
        identifiers = self.entity_identifier_columns(parent)
        equivalent = [
            fk
            for fk in candidates
            if resolved_column in identifiers
            and fk.ref_columns
            and all(
                self.resolve_column_name(parent, column) in identifiers
                for column in fk.ref_columns
            )
        ]
        return equivalent[0] if len(equivalent) == 1 else None

    def has_join_path(self, left: str, right: str) -> bool:
        return self.join_path(left, right) is not None

    def join_path(self, left: str, right: str) -> list[str] | None:
        if left == right:
            return [left]
        graph: dict[str, set[str]] = {name: set() for name in self.tables}
        for table in self.tables.values():
            for fk in table.foreign_keys:
                graph[table.name].add(fk.ref_table)
                graph[fk.ref_table].add(table.name)
        queue: deque[tuple[str, list[str]]] = deque([(left, [left])])
        seen = {left}
        while queue:
            node, path = queue.popleft()
            for nxt in graph[node]:
                if nxt in seen:
                    continue
                if nxt == right:
                    return path + [nxt]
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
        return None


def sqlite_type(type_name: str) -> str:
    t = type_name.lower()
    if any(x in t for x in ("int", "decimal", "numeric", "number", "real", "float", "double")):
        return "REAL" if not t.startswith("int") else "INTEGER"
    return "TEXT"


def default_value(table: str, column: str, type_name: str, index: int = 1) -> Any:
    name = column.lower()
    if any(x in type_name.lower() for x in ("int", "decimal", "numeric", "number", "real", "float", "double")):
        if name.endswith("_id") or name.endswith("id"):
            return index
        return index
    if name.endswith("_id") or name.endswith("id"):
        return f"{table[:1].upper()}{index:03d}"
    if "region" in name:
        return "East"
    if "city" in name:
        return "Shanghai"
    if "status" in name:
        return "paid"
    if any(x in name for x in ("date", "time", "month")):
        return f"2025-06-{index:02d}"
    return f"{column}_{index}"
