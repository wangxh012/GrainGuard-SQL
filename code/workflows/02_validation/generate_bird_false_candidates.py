from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.contract import MetricContract
from grain_test_sql.pipeline import run_pipeline
from grain_test_sql.schema_registry import SchemaRegistry
from grain_test_sql.bird_utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="artifacts/datasets/bird/eval/bird_eval_set.jsonl")
    parser.add_argument("--auto-contracts", default="", help="Optional JSONL from generate_bird_auto_contracts.py")
    parser.add_argument(
        "--auto-contract-row-set",
        action="store_true",
        help="Use only eval rows present in --auto-contracts, preserving that file order before --limit.",
    )
    parser.add_argument("--contract-source", choices=["silver", "auto"], default="silver")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="artifacts/runs/bird/shared/01_gold_fault_validation/false_candidates_from_gold.jsonl")
    parser.add_argument(
        "--contract-confidence",
        choices=["all", "high"],
        default="high",
        help="Select which silver contracts are eligible for mutation.",
    )
    parser.add_argument(
        "--mutation-mode",
        choices=["gold-sql", "contract-synthetic", "gold-sql-with-fallback"],
        default="gold-sql-with-fallback",
        help=(
            "gold-sql mutates the BIRD gold SQL AST in place; contract-synthetic keeps the older "
            "minimal SQL builder; gold-sql-with-fallback tries gold-sql first and then falls back."
        ),
    )
    parser.add_argument(
        "--detectable-only",
        action="store_true",
        help="For gold-sql mutations, keep only shapes that the current counterfactual witness can validate robustly.",
    )
    parser.add_argument(
        "--gold-sanity-results",
        dest="pass_results",
        default="",
        help="Optional legacy alias for --pass-results.",
    )
    parser.add_argument(
        "--pass-results",
        dest="pass_results",
        default="",
        help="Optional validation JSONL used to filter mutation candidates to rows whose gold SQL passed.",
    )
    parser.add_argument(
        "--require-gold-pass",
        action="store_true",
        help="Only mutate contract rows whose gold SQL validation result is PASS in --pass-results.",
    )
    parser.add_argument(
        "--force-synthetic-fallback",
        action="store_true",
        help="If normal mutation fails, emit a marked synthetic invariant fault for supported contract types.",
    )
    parser.add_argument(
        "--prefer-validated-violation",
        action="store_true",
        help=(
            "Build all available false-candidate variants and keep the first one that the local "
            "contract witness already classifies as VIOLATION."
        ),
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.eval_set))
    auto_contract_rows = read_jsonl(Path(args.auto_contracts)) if args.auto_contracts else []
    gold_pass_keys: set[tuple[Any, ...]] = set()
    if args.require_gold_pass:
        if not args.pass_results:
            raise SystemExit("--require-gold-pass requires --pass-results")
        gold_pass_keys = load_validation_pass_keys(Path(args.pass_results))
    if args.auto_contract_row_set:
        rows = filter_eval_rows_by_contract_order(rows, auto_contract_rows)
    if args.limit:
        rows = rows[: args.limit]
    auto_contracts = contracts_by_key(auto_contract_rows, "auto_contracts") if auto_contract_rows else {}
    predictions: list[dict[str, Any]] = []
    stats = {
        "rows": len(rows),
        "eligible_contracts": 0,
        "generated_false_candidates": 0,
        "gold_sql_mutations": 0,
        "contract_synthetic_mutations": 0,
        "forced_synthetic_mutations": 0,
        "det_unsafe_count": 0,
        "cgr_unsafe_count": 0,
        "grt_unsafe_sum": 0,
        "art_unsafe_avg": 0,
        "est_unsafe_avg_event": 0,
        "tst_unsafe_topk": 0,
        "skipped_no_mutation": 0,
        "skipped_low_confidence_contracts": 0,
        "skipped_gold_sanity_not_pass": 0,
        "validated_candidate_violations": 0,
        "validated_candidate_nonviolations": 0,
    }

    for row in rows:
        key = (row["question_id"], row["db_id"])
        if args.contract_source == "auto":
            contracts = auto_contracts.get(key) or row.get("auto_contracts") or []
        else:
            contracts = row.get("silver_contracts") or []
        if args.contract_source == "silver" and args.contract_confidence == "high":
            before = len(contracts)
            contracts = [contract for contract in contracts if is_high_confidence_silver(contract)]
            stats["skipped_low_confidence_contracts"] += before - len(contracts)
        for contract_index, contract in enumerate(contracts):
            stats["eligible_contracts"] += 1
            contract_key = (
                row["question_id"],
                row["db_id"],
                contract.get("contract_type"),
                contract_index,
            )
            if args.require_gold_pass and contract_key not in gold_pass_keys:
                stats["skipped_gold_sanity_not_pass"] += 1
                continue
            registry = SchemaRegistry.from_yaml(row["schema_registry_path"])
            candidate_options = build_candidate_options(
                row,
                contract,
                registry,
                args.mutation_mode,
                args.detectable_only,
                args.force_synthetic_fallback,
            )
            selected = select_candidate_option(
                candidate_options,
                contract,
                registry,
                prefer_validated_violation=args.prefer_validated_violation,
            )
            if not selected:
                stats["skipped_no_mutation"] += 1
                continue
            candidate_sql = selected["candidate_sql"]
            mutation_origin = selected["source"]
            validation_status = selected.get("generation_validation_status")
            if args.prefer_validated_violation:
                if validation_status == "VIOLATION":
                    stats["validated_candidate_violations"] += 1
                else:
                    stats["validated_candidate_nonviolations"] += 1
            if mutation_origin == "gold_sql_fault_injection":
                stats["gold_sql_mutations"] += 1
            elif mutation_origin == "contract_synthetic_negative":
                stats["contract_synthetic_mutations"] += 1
            elif mutation_origin == "contract_synthetic_forced":
                stats["forced_synthetic_mutations"] += 1
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "db_id": row["db_id"],
                    "candidate_sql": candidate_sql,
                    "candidate sql": candidate_sql,
                    "mutation": mutation_name(contract["contract_type"]),
                    "source": mutation_origin,
                    "gold_sql": row["gold_sql"],
                    "contract_type": contract["contract_type"],
                    "contract_index": contract_index,
                    "generation_validation_status": validation_status,
                }
            )
            stats["generated_false_candidates"] += 1
            if contract["contract_type"] == "DET":
                stats["det_unsafe_count"] += 1
            elif contract["contract_type"] == "CGR":
                stats["cgr_unsafe_count"] += 1
            elif contract["contract_type"] == "GRT":
                stats["grt_unsafe_sum"] += 1
            elif contract["contract_type"] == "ART":
                stats["art_unsafe_avg"] += 1
            elif contract["contract_type"] == "EST":
                stats["est_unsafe_avg_event"] += 1
            elif contract["contract_type"] == "TST":
                stats["tst_unsafe_topk"] += 1

    write_jsonl(Path(args.out), predictions)
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_validation_pass_keys(path: Path) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    for row in read_jsonl(path):
        grain_result = row.get("grain_result") or {}
        if grain_result.get("status") != "PASS":
            continue
        if row.get("question_id") is None or row.get("db_id") is None or row.get("contract_type") is None:
            continue
        keys.add(
            (
                row["question_id"],
                row["db_id"],
                row["contract_type"],
                row.get("contract_index", 0),
            )
        )
    return keys


def build_candidate_options(
    row: dict[str, Any],
    contract: dict[str, Any],
    registry: SchemaRegistry,
    mutation_mode: str,
    detectable_only: bool,
    force_synthetic_fallback: bool,
) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    sql_tables = set(row.get("sql_tables") or [])

    if mutation_mode in {"gold-sql", "gold-sql-with-fallback"}:
        candidate_sql = inject_gold_sql_fault(row["gold_sql"], contract, registry, sql_tables)
        if candidate_sql and (
            not detectable_only or is_detectable_gold_fault_shape(candidate_sql, contract)
        ):
            options.append(
                {
                    "candidate_sql": candidate_sql,
                    "source": "gold_sql_fault_injection",
                }
            )

    fallback_builders: list[tuple[str, Any]] = []
    if detectable_only and force_synthetic_fallback:
        # Forced synthetic candidates are explicitly shaped for the synthetic witness and
        # have much higher empirical violation yield than the older natural fallback.
        fallback_builders.append(("contract_synthetic_forced", build_forced_synthetic_candidate))
    if mutation_mode in {"contract-synthetic", "gold-sql-with-fallback"}:
        fallback_builders.append(("contract_synthetic_negative", build_false_candidate))
    if force_synthetic_fallback and not any(source == "contract_synthetic_forced" for source, _ in fallback_builders):
        fallback_builders.append(("contract_synthetic_forced", build_forced_synthetic_candidate))

    seen_sql = {option["candidate_sql"] for option in options}
    for source, builder in fallback_builders:
        if source == "contract_synthetic_negative":
            candidate_sql = builder(contract, registry, sql_tables)
        else:
            candidate_sql = builder(contract, registry)
        if not candidate_sql or candidate_sql in seen_sql:
            continue
        options.append({"candidate_sql": candidate_sql, "source": source})
        seen_sql.add(candidate_sql)
    return options


def select_candidate_option(
    options: list[dict[str, Any]],
    contract: dict[str, Any],
    registry: SchemaRegistry,
    *,
    prefer_validated_violation: bool,
) -> dict[str, Any] | None:
    if not options:
        return None
    if not prefer_validated_violation:
        return dict(options[0])

    metric_contract = MetricContract.from_dict(contract["contract"])
    first_validated: dict[str, Any] | None = None
    for option in options:
        selected = dict(option)
        try:
            status = run_pipeline(selected["candidate_sql"], registry, metric_contract).status
        except Exception as exc:
            status = f"ERROR:{type(exc).__name__}"
        selected["generation_validation_status"] = status
        if first_validated is None:
            first_validated = selected
        if status == "VIOLATION":
            return selected
    return first_validated


def contracts_by_key(rows: list[dict[str, Any]], field: str) -> dict[tuple[int, str], list[dict[str, Any]]]:
    contracts: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        contracts[(row["question_id"], row["db_id"])] = row.get(field) or []
    return contracts


def filter_eval_rows_by_contract_order(
    eval_rows: list[dict[str, Any]],
    contract_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eval_by_key = {
        (row["question_id"], row["db_id"]): row
        for row in eval_rows
    }
    return [
        eval_by_key[(row["question_id"], row["db_id"])]
        for row in contract_rows
        if (row.get("question_id"), row.get("db_id")) in eval_by_key
    ]


def is_high_confidence_silver(contract: dict[str, Any]) -> bool:
    tier = contract.get("silver_tier")
    if tier:
        return tier == "high"
    return not contract.get("confidence", "").startswith("weak_")


def inject_gold_sql_fault(
    gold_sql: str,
    contract: dict[str, Any],
    registry: SchemaRegistry,
    sql_tables: set[str],
) -> str | None:
    try:
        tree = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return None

    contract_type = contract.get("contract_type")
    body = contract.get("contract") or {}
    measures = body.get("measures") or []
    if not measures:
        return None
    measure = measures[0]
    original = tree.sql(dialect="sqlite")

    if contract_type == "DET":
        mutated = mutate_count_distinct(tree, matching_refs_for_det(measure, registry), prefer_star=True)
        if not mutated:
            return None
        return changed_sql(tree, original)

    if contract_type == "CGR":
        qualify_unqualified_measure_columns(tree, measure)
        mutated_count = mutate_count_distinct(tree, matching_refs_for_grain(measure), prefer_star=False)
        target = choose_replication_table(measure["source_table"], registry, sql_tables)
        mutated_join = add_lower_grain_left_join(tree, measure["source_table"], target, registry) if target else False
        if not (mutated_count or mutated_join):
            return None
        return changed_sql(tree, original)

    if contract_type == "GRT":
        return inject_lower_grain_aggregate_fault(tree, original, "SUM", measure, registry, sql_tables)

    if contract_type == "ART":
        return inject_lower_grain_aggregate_fault(tree, original, "AVG", measure, registry, sql_tables)

    if contract_type == "EST":
        return inject_est_fault(tree, original, measure)

    return None


def is_detectable_gold_fault_shape(sql: str, contract: dict[str, Any]) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return False
    where = tree.find(exp.Where)
    if where and (where.find(exp.Subquery) or len(list(where.find_all(exp.Select))) > 0):
        return False

    contract_type = contract.get("contract_type")
    measures = ((contract.get("contract") or {}).get("measures") or [])
    if not measures:
        return False
    measure = measures[0]
    if contract_type == "DET":
        return True
    if contract_type == "CGR":
        return has_direct_projection_aggregate(tree, "COUNT", matching_refs_for_grain(measure))
    if contract_type == "GRT":
        source_ref = measure_source_ref(measure)
        return bool(source_ref and has_direct_projection_aggregate(tree, "SUM", {source_ref}))
    if contract_type == "ART":
        source_ref = measure_source_ref(measure)
        return bool(source_ref and has_direct_projection_aggregate(tree, "AVG", {source_ref}))
    if contract_type == "EST":
        source_ref = measure_source_ref(measure)
        return bool(source_ref and has_direct_projection_aggregate(tree, "AVG", {source_ref}))
    return False


def has_direct_projection_aggregate(tree: exp.Expression, function: str, accepted_refs: set[str]) -> bool:
    aliases = table_aliases(tree)
    select = tree.find(exp.Select)
    if not select:
        return False
    aggregate_type = {"COUNT": exp.Count, "SUM": exp.Sum, "AVG": exp.Avg}.get(function)
    if not aggregate_type:
        return False
    for projection in select.expressions:
        expr = projection.this if isinstance(projection, exp.Alias) else projection
        expr = expr.this if isinstance(expr, exp.Cast) else expr
        if not isinstance(expr, aggregate_type):
            continue
        ref = normalized_first_column_ref(expr, aliases)
        if function == "COUNT" and isinstance(expr.this, exp.Star):
            return True
        if not accepted_refs or ref_matches_any(ref or "", accepted_refs):
            return True
    return False


def inject_lower_grain_aggregate_fault(
    tree: exp.Expression,
    original: str,
    function: str,
    measure: dict[str, Any],
    registry: SchemaRegistry,
    sql_tables: set[str],
) -> str | None:
    source_ref = measure_source_ref(measure)
    if not source_ref or not has_aggregate_on_ref(tree, function, source_ref):
        return None
    qualify_unqualified_measure_columns(tree, measure)
    target = choose_replication_table(measure["source_table"], registry, sql_tables)
    if not target:
        return None
    if not add_lower_grain_left_join(tree, measure["source_table"], target, registry):
        return None
    return changed_sql(tree, original)


def inject_est_fault(tree: exp.Expression, original: str, measure: dict[str, Any]) -> str | None:
    source_column = measure.get("source_column")
    event_table = measure.get("event_table") or measure.get("source_table")
    if not source_column or not event_table:
        return None
    aliases = table_aliases(tree)
    event_alias = first_alias_for_table(aliases, event_table)
    if not event_alias:
        return None
    replacement = exp.column(source_column, table=event_alias)
    for avg in tree.find_all(exp.Avg):
        current_ref = normalized_first_column_ref(avg, aliases)
        if current_ref and current_ref.lower() == f"{event_table}.{source_column}".lower():
            continue
        avg.set("this", replacement.copy())
        return changed_sql(tree, original)
    return None


def mutate_count_distinct(tree: exp.Expression, accepted_refs: set[str], prefer_star: bool) -> bool:
    aliases = table_aliases(tree)
    for count in tree.find_all(exp.Count):
        distinct = count.this if isinstance(count.this, exp.Distinct) else None
        if not distinct or not distinct.expressions:
            continue
        argument = distinct.expressions[0]
        ref = normalized_column_ref(argument, aliases)
        if accepted_refs and ref and not ref_matches_any(ref, accepted_refs):
            continue
        if prefer_star:
            count.set("this", exp.Star())
        else:
            count.set("this", argument.copy())
        return True
    return False


def qualify_unqualified_measure_columns(tree: exp.Expression, measure: dict[str, Any]) -> None:
    source_table = measure.get("source_table")
    source_column = measure.get("source_column")
    if not source_table or not source_column:
        return
    aliases = table_aliases(tree)
    source_alias = first_alias_for_table(aliases, source_table)
    if not source_alias:
        return
    accepted_columns = {source_column.lower()}
    for grain_ref in measure.get("base_grain") or []:
        if "." in grain_ref:
            accepted_columns.add(grain_ref.rsplit(".", 1)[1].lower())
    for agg in tree.find_all(exp.Count, exp.Sum, exp.Avg):
        for col in list(agg.find_all(exp.Column)):
            if col.table or col.name.lower() not in accepted_columns:
                continue
            col.replace(exp.column(col.name, table=source_alias))


def add_lower_grain_left_join(
    tree: exp.Expression,
    source_table: str,
    target_table: str | None,
    registry: SchemaRegistry,
) -> bool:
    if not target_table or target_table not in registry.tables:
        return False
    join_pairs = replication_join_pairs(source_table, target_table, registry)
    if not join_pairs:
        return False
    aliases = table_aliases(tree)
    source_alias = first_alias_for_table(aliases, source_table)
    if not source_alias:
        return False
    existing_aliases = set(aliases)
    child_alias = unique_alias(target_table, existing_aliases)
    conditions = [
        sqlglot.parse_one(
            f"{quote_ident(child_alias)}.{quote_ident(target_col)} = "
            f"{quote_ident(source_alias)}.{quote_ident(source_col)}",
            read="sqlite",
        )
        for target_col, source_col in join_pairs
    ]
    on_expr = conditions[0]
    for condition in conditions[1:]:
        on_expr = exp.and_(on_expr, condition)
    table = exp.Table(
        this=exp.to_identifier(target_table, quoted=True),
        alias=exp.TableAlias(this=exp.to_identifier(child_alias, quoted=True)),
    )
    join = exp.Join(this=table, on=on_expr, side="LEFT")
    select = tree.find(exp.Select)
    if not select:
        return False
    joins = list(select.args.get("joins") or [])
    joins.append(join)
    select.set("joins", joins)
    return True


def replication_join_pairs(
    source_table: str,
    target_table: str,
    registry: SchemaRegistry,
) -> list[tuple[str, str]]:
    direct_fk = registry.fk_to_parent(target_table, source_table)
    if direct_fk and direct_fk.many_allowed and not registry.child_is_unique_for_parent(target_table, source_table, direct_fk):
        return list(zip(direct_fk.columns, direct_fk.ref_columns))

    source_fks = [
        fk
        for fk in registry.tables[source_table].foreign_keys
        if fk.many_allowed
    ]
    for target_fk in registry.tables[target_table].foreign_keys:
        if not target_fk.many_allowed:
            continue
        for source_fk in source_fks:
            same_parent = (
                target_fk.ref_table == source_fk.ref_table
                and target_fk.ref_columns == source_fk.ref_columns
            )
            if same_parent and not registry.sibling_is_unique_for_source(target_table, source_table, target_fk):
                return list(zip(target_fk.columns, source_fk.columns))
    return []


def choose_replication_table(source_table: str, registry: SchemaRegistry, sql_tables: set[str]) -> str | None:
    candidates: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for child, fk in registry.children_of(source_table):
        if child == source_table or not fk.many_allowed:
            continue
        if registry.child_is_unique_for_parent(child, source_table, fk):
            continue
        candidates.append((child, child in sql_tables))
        seen.add(child)

    source_parents = {
        (fk.ref_table, tuple(fk.ref_columns))
        for fk in registry.tables[source_table].foreign_keys
        if fk.many_allowed
    }
    for table_name, table in registry.tables.items():
        if table_name == source_table or table_name in seen:
            continue
        for fk in table.foreign_keys:
            if not fk.many_allowed:
                continue
            if (fk.ref_table, tuple(fk.ref_columns)) not in source_parents:
                continue
            if registry.sibling_is_unique_for_source(table_name, source_table, fk):
                continue
            candidates.append((table_name, table_name in sql_tables))
            seen.add(table_name)
            break

    for table, already_scoped in candidates:
        if already_scoped:
            return table
    if candidates:
        return candidates[0][0]
    return None


def table_aliases(tree: exp.Expression) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        aliases[table.alias_or_name] = table.name
        aliases[table.name] = table.name
    return aliases


def first_alias_for_table(aliases: dict[str, str], table: str) -> str | None:
    for alias, resolved in aliases.items():
        if resolved.lower() == table.lower():
            return alias
    return None


def unique_alias(table: str, existing_aliases: set[str]) -> str:
    base = f"grain_fault_{safe_alias_part(table)}"
    alias = base
    suffix = 1
    while alias in existing_aliases:
        suffix += 1
        alias = f"{base}_{suffix}"
    return alias


def safe_alias_part(table: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in table.lower()).strip("_") or "t"


def has_aggregate_on_ref(tree: exp.Expression, function: str, source_ref: str) -> bool:
    aliases = table_aliases(tree)
    aggregate_types: tuple[type[exp.Expression], ...]
    if function == "SUM":
        aggregate_types = (exp.Sum,)
    elif function == "AVG":
        aggregate_types = (exp.Avg,)
    else:
        return False
    return any(
        ref_matches(normalized_first_column_ref(agg, aliases), source_ref)
        for agg in tree.find_all(*aggregate_types)
    )


def matching_refs_for_det(measure: dict[str, Any], registry: SchemaRegistry) -> set[str]:
    refs = set()
    entity_key = measure.get("entity_key")
    event_table = measure.get("event_table")
    if entity_key:
        refs.add(entity_key)
    if entity_key and event_table and "." in entity_key:
        entity_table, _ = entity_key.split(".", 1)
        fk = registry.fk_to_parent(event_table, entity_table) if event_table in registry.tables else None
        if fk:
            refs.update(f"{event_table}.{column}" for column in fk.columns)
    return refs


def matching_refs_for_grain(measure: dict[str, Any]) -> set[str]:
    refs = set(measure.get("base_grain") or [])
    source_ref = measure_source_ref(measure)
    if source_ref:
        refs.add(source_ref)
    return refs


def measure_source_ref(measure: dict[str, Any]) -> str | None:
    if not measure.get("source_table") or not measure.get("source_column"):
        return None
    return f'{measure["source_table"]}.{measure["source_column"]}'


def normalized_first_column_ref(node: exp.Expression, aliases: dict[str, str]) -> str | None:
    for col in node.find_all(exp.Column):
        return normalized_column_ref(col, aliases)
    return None


def normalized_column_ref(node: exp.Expression, aliases: dict[str, str]) -> str | None:
    if not isinstance(node, exp.Column):
        return None
    if not node.table:
        return node.name
    return f"{aliases.get(node.table, node.table)}.{node.name}"


def ref_matches_any(ref: str, accepted_refs: set[str]) -> bool:
    return any(ref_matches(ref, accepted) for accepted in accepted_refs)


def ref_matches(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    if left.lower() == right.lower():
        return True
    left_col = left.rsplit(".", 1)[-1].lower()
    right_col = right.rsplit(".", 1)[-1].lower()
    return left_col == right_col


def changed_sql(tree: exp.Expression, original: str) -> str | None:
    mutated = tree.sql(dialect="sqlite")
    return mutated if mutated != original else None


def build_false_candidate(contract: dict[str, Any], registry: SchemaRegistry, sql_tables: set[str]) -> str | None:
    body = contract["contract"]
    measure = body["measures"][0]
    contract_type = contract.get("contract_type")
    if contract_type == "DET":
        return build_det_false_candidate(measure, registry)
    if contract_type == "CGR":
        return build_cgr_false_candidate(measure, registry, sql_tables)
    if contract_type in {"GRT", "ART"}:
        return build_grain_measure_false_candidate(contract_type, measure, registry, sql_tables)
    if contract_type == "EST":
        return build_est_false_candidate(measure, registry)
    if contract_type == "TST":
        return build_tst_false_candidate(body, measure, registry)
    return None


def build_forced_synthetic_candidate(contract: dict[str, Any], registry: SchemaRegistry) -> str | None:
    body = contract.get("contract") or {}
    measures = body.get("measures") or []
    if not measures:
        return None
    measure = measures[0]
    contract_type = contract.get("contract_type")
    if contract_type == "DET":
        return build_forced_det_candidate(measure, registry)
    if contract_type == "CGR":
        return build_forced_cgr_candidate(measure, registry)
    if contract_type == "GRT":
        return build_forced_grt_candidate(measure, registry)
    if contract_type == "ART":
        return build_forced_art_candidate(measure, registry)
    if contract_type == "EST":
        return build_forced_est_candidate(measure, registry)
    if contract_type == "TST":
        return build_forced_tst_candidate(body, measure, registry)
    return None


def build_forced_det_candidate(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    entity_key = measure.get("entity_key")
    if not entity_key:
        return None
    entity_table, entity_col = entity_key.split(".", 1)
    if entity_table not in registry.tables or entity_col not in registry.tables[entity_table].columns:
        return None
    fault = synthetic_entity_fault_column(entity_table, entity_col, registry)
    literal = sqlite_literal(synthetic_trigger_value(registry, entity_table, fault))
    return (
        "/* grain_fault:synthetic */ "
        f"SELECT COUNT({quote_ref('e', entity_col)}) + "
        f"SUM(CASE WHEN {quote_ref('e', fault)} = {literal} THEN 1 ELSE 0 END) AS entity_count "
        f"FROM {quote_ident(entity_table)} AS e"
    )


def build_forced_cgr_candidate(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    source_table = measure.get("source_table")
    if not source_table or source_table not in registry.tables:
        return None
    count_col = forced_grain_column(measure, registry)
    fault = synthetic_fault_column(measure, registry) or count_col
    if not count_col or not fault:
        return None
    literal = sqlite_literal(synthetic_trigger_value(registry, source_table, fault))
    return (
        "/* grain_fault:synthetic */ "
        f"SELECT COUNT({quote_ref('s', count_col)}) + "
        f"SUM(CASE WHEN {quote_ref('s', fault)} = {literal} THEN 1 ELSE 0 END) AS grain_count "
        f"FROM {quote_ident(source_table)} AS s"
    )


def build_forced_grt_candidate(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    source_table = measure.get("source_table")
    source_column = measure.get("source_column")
    if not source_table or not source_column or source_table not in registry.tables:
        return None
    fault = synthetic_fault_column(measure, registry)
    measure_ref = quote_ref("s", source_column)
    if not fault:
        return (
            "/* grain_fault:synthetic */ "
            f"SELECT SUM({measure_ref}) + COUNT(*) AS measure_value "
            f"FROM {quote_ident(source_table)} AS s"
        )
    literal = sqlite_literal(synthetic_trigger_value(registry, source_table, fault))
    return (
        "/* grain_fault:synthetic */ "
        f"SELECT SUM({measure_ref}) + "
        f"SUM(CASE WHEN {quote_ref('s', fault)} = {literal} THEN {measure_ref} ELSE 0 END) AS measure_value "
        f"FROM {quote_ident(source_table)} AS s"
    )


def build_forced_art_candidate(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    source_table = measure.get("source_table")
    source_column = measure.get("source_column")
    if not source_table or not source_column or source_table not in registry.tables:
        return None
    fault = synthetic_fault_column(measure, registry)
    if not fault:
        return None
    literal = sqlite_literal(synthetic_trigger_value(registry, source_table, fault))
    measure_ref = quote_ref("s", source_column)
    return (
        "/* grain_fault:synthetic */ "
        f"SELECT AVG({measure_ref} + "
        f"CASE WHEN {quote_ref('s', fault)} = {literal} THEN {measure_ref} ELSE 0 END) AS measure_value "
        f"FROM {quote_ident(source_table)} AS s"
    )


def build_forced_est_candidate(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    source_table = measure.get("event_table") or measure.get("source_table")
    if not source_table or source_table not in registry.tables:
        return None
    return (
        "/* grain_fault:synthetic */ "
        f"SELECT COUNT(*) AS avg_entity_total FROM {quote_ident(source_table)} AS ev"
    )


def build_forced_tst_candidate(
    contract_body: dict[str, Any],
    measure: dict[str, Any],
    registry: SchemaRegistry,
) -> str | None:
    group = contract_body.get("output_group") or {}
    group_table = group.get("table")
    group_col = group.get("column")
    source_table = measure.get("source_table")
    if (
        not group_table
        or not group_col
        or not source_table
        or group_table not in registry.tables
        or source_table not in registry.tables
        or group_col not in registry.tables[group_table].columns
    ):
        return None
    if source_table == group_table:
        from_clause = f"FROM {quote_ident(source_table)} AS s"
    else:
        join = join_condition(source_table, "s", group_table, "g", registry)
        if join:
            from_clause = (
                f"FROM {quote_ident(group_table)} AS g "
                f"INNER JOIN {quote_ident(source_table)} AS s ON {join}"
            )
        else:
            join = join_condition(group_table, "g", source_table, "s", registry)
            if not join:
                return None
            from_clause = (
                f"FROM {quote_ident(source_table)} AS s "
                f"INNER JOIN {quote_ident(group_table)} AS g ON {join}"
            )
    group_alias = "s" if source_table == group_table else "g"
    if measure.get("source_column"):
        metric = f"SUM(s.{quote_ident(measure['source_column'])})"
    else:
        metric = "COUNT(*)"
    return (
        "/* grain_fault:synthetic */ "
        f"SELECT {group_alias}.{quote_ident(group_col)} AS top_group, "
        f"{metric} AS ranking_metric {from_clause} "
        f"GROUP BY {group_alias}.{quote_ident(group_col)} "
        f"ORDER BY ranking_metric ASC LIMIT 1"
    )


def build_det_false_candidate(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    if not measure.get("entity_key") or not measure.get("event_table"):
        return None
    entity_table, entity_col = measure["entity_key"].split(".", 1)
    event_table = measure["event_table"]
    fk = registry.fk_to_parent(event_table, entity_table)
    join = join_condition(event_table, "ev", entity_table, "e", registry)
    if not join:
        return None
    if fk and set(fk.columns) == set(registry.tables[event_table].primary_key):
        return None
    count_ref = f'ev.{quote_ident(fk.columns[0])}' if fk else f'e.{quote_ident(entity_col)}'
    return (
        f"SELECT COUNT({count_ref}) AS entity_count "
        f'FROM {quote_ident(entity_table)} AS e '
        f'INNER JOIN {quote_ident(event_table)} AS ev ON {join}'
    )


def build_grain_measure_false_candidate(
    contract_type: str,
    measure: dict[str, Any],
    registry: SchemaRegistry,
    sql_tables: set[str],
) -> str | None:
    source_table = measure["source_table"]
    source_column = measure.get("source_column")
    if not source_column:
        return None
    child_table = choose_child_table(source_table, registry, sql_tables)
    if not child_table:
        return None
    join = join_condition(child_table, "ev", source_table, "e", registry)
    if not join:
        return None
    function = "SUM" if contract_type == "GRT" else "AVG"
    return (
        f'SELECT {function}(e.{quote_ident(source_column)}) AS measure_value '
        f'FROM {quote_ident(source_table)} AS e '
        f'INNER JOIN {quote_ident(child_table)} AS ev ON {join}'
    )


def build_cgr_false_candidate(measure: dict[str, Any], registry: SchemaRegistry, sql_tables: set[str]) -> str | None:
    source_table = measure["source_table"]
    base_grain = measure.get("base_grain") or []
    if not base_grain or "." not in base_grain[0]:
        return None
    _, grain_col = base_grain[0].split(".", 1)
    child_table = choose_child_table(source_table, registry, sql_tables)
    if not child_table:
        return None
    join = join_condition(child_table, "ev", source_table, "e", registry)
    if not join:
        return None
    return (
        f'SELECT COUNT(e.{quote_ident(grain_col)}) AS grain_count '
        f'FROM {quote_ident(source_table)} AS e '
        f'INNER JOIN {quote_ident(child_table)} AS ev ON {join}'
    )


def build_est_false_candidate(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    if not measure.get("entity_key") or not measure.get("event_table") or not measure.get("source_column"):
        return None
    entity_table, _ = measure["entity_key"].split(".", 1)
    event_table = measure["event_table"]
    if event_table == entity_table:
        return None
    join = join_condition(event_table, "ev", entity_table, "e", registry)
    if not join:
        return None
    fk = registry.fk_to_parent(event_table, entity_table)
    if fk and set(fk.columns) == set(registry.tables[event_table].primary_key):
        return None
    return (
        f'SELECT AVG(ev.{quote_ident(measure["source_column"])}) AS avg_entity_total '
        f'FROM {quote_ident(entity_table)} AS e '
        f'INNER JOIN {quote_ident(event_table)} AS ev ON {join}'
    )


def build_tst_false_candidate(
    contract_body: dict[str, Any],
    measure: dict[str, Any],
    registry: SchemaRegistry,
) -> str | None:
    group = contract_body.get("output_group") or {}
    group_table = group.get("table")
    group_col = group.get("column")
    source_table = measure.get("source_table")
    if not group_table or not group_col or not source_table:
        return None
    if source_table == group_table:
        from_clause = f'FROM {quote_ident(source_table)} AS s'
    else:
        join = join_condition(source_table, "s", group_table, "g", registry)
        if join:
            from_clause = (
                f'FROM {quote_ident(group_table)} AS g '
                f'INNER JOIN {quote_ident(source_table)} AS s ON {join}'
            )
        else:
            join = join_condition(group_table, "g", source_table, "s", registry)
            if not join:
                return None
            from_clause = (
                f'FROM {quote_ident(source_table)} AS s '
                f'INNER JOIN {quote_ident(group_table)} AS g ON {join}'
            )
    group_alias = "s" if source_table == group_table else "g"
    metric = "COUNT(*)" if not measure.get("source_column") else f'SUM(s.{quote_ident(measure["source_column"])})'
    return (
        f'SELECT {group_alias}.{quote_ident(group_col)} AS top_group, {metric} AS ranking_metric '
        f'{from_clause} '
        f'GROUP BY {group_alias}.{quote_ident(group_col)} '
        f'ORDER BY ranking_metric ASC LIMIT 1'
    )


def choose_child_table(source_table: str, registry: SchemaRegistry, sql_tables: set[str]) -> str | None:
    children = [
        (child, fk)
        for child, fk in registry.children_of(source_table)
        if child != source_table and fk.many_allowed
    ]
    duplicable = [
        (child, fk)
        for child, fk in children
        if set(fk.columns) != set(registry.tables[child].primary_key)
    ]
    for child, _ in duplicable:
        if child in sql_tables:
            return child
    if duplicable:
        return duplicable[0][0]
    return None


def join_condition(child: str, child_alias: str, parent: str, parent_alias: str, registry: SchemaRegistry) -> str | None:
    fk = registry.fk_to_parent(child, parent)
    if not fk:
        return None
    parts = [
        f"{child_alias}.{quote_ident(child_col)} = {parent_alias}.{quote_ident(parent_col)}"
        for child_col, parent_col in zip(fk.columns, fk.ref_columns)
    ]
    return " AND ".join(parts)


def forced_grain_column(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    for grain_ref in measure.get("base_grain") or []:
        if "." not in grain_ref:
            continue
        table, column = grain_ref.split(".", 1)
        if table == measure.get("source_table") and column in registry.tables[table].columns:
            return column
    return measure.get("source_column")


def synthetic_fault_column(measure: dict[str, Any], registry: SchemaRegistry) -> str | None:
    source_table = measure.get("source_table")
    if not source_table or source_table not in registry.tables:
        return None
    schema = registry.tables[source_table]
    blocked = {measure.get("source_column")}
    blocked.update(ref.split(".", 1)[1] for ref in measure.get("base_grain") or [] if "." in ref)
    foreign_key_columns = {
        column
        for fk in schema.foreign_keys
        for column in fk.columns
    }
    unique_columns = {
        column
        for columns in [schema.primary_key, *schema.unique]
        for column in columns
    }
    candidates = [
        col
        for col in schema.columns
        if col not in blocked
        and col not in foreign_key_columns
        and col not in unique_columns
    ]
    if not candidates:
        candidates = [
            col
            for col in schema.columns
            if col not in blocked and col not in unique_columns
        ]
    return candidates[0] if candidates else None


def synthetic_entity_fault_column(entity_table: str, entity_col: str, registry: SchemaRegistry) -> str:
    schema = registry.tables[entity_table]
    foreign_key_columns = {
        column
        for fk in schema.foreign_keys
        for column in fk.columns
    }
    unique_columns = {
        column
        for columns in [schema.primary_key, *schema.unique]
        for column in columns
    }
    candidates = [
        col
        for col in schema.columns
        if col != entity_col
        and col not in foreign_key_columns
        and col not in unique_columns
    ]
    if not candidates:
        candidates = [
            col
            for col in schema.columns
            if col != entity_col and col not in unique_columns
        ]
    return candidates[0] if candidates else entity_col


def synthetic_trigger_value(registry: SchemaRegistry, table: str, column: str) -> Any:
    typ = registry.tables[table].columns[column].lower()
    if "date" in typ or "time" in typ:
        return "2099-01-01"
    if any(x in typ for x in ("int", "decimal", "numeric", "real", "float", "double")):
        return 999001
    return "__grain_fault_trigger__"


def sqlite_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def quote_ref(alias: str, column: str) -> str:
    return f"{quote_ident(alias)}.{quote_ident(column)}"


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def mutation_name(contract_type: str) -> str:
    if contract_type == "DET":
        return "unsafe_count_without_distinct"
    if contract_type == "CGR":
        return "unsafe_count_after_lower_grain_join"
    if contract_type == "GRT":
        return "unsafe_sum_after_lower_grain_join"
    if contract_type == "ART":
        return "unsafe_avg_after_lower_grain_join"
    if contract_type == "EST":
        return "unsafe_avg_event_value"
    if contract_type == "TST":
        return "unsafe_reverse_topk_metric"
    return "unsupported_mutation"


if __name__ == "__main__":
    main()
