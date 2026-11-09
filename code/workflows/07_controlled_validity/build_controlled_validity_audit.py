from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.contract import MetricContract
from grain_test_sql.executor import execute_sql
from grain_test_sql.pair_builder import PairBuilder, TestPair
from grain_test_sql.pipeline import run_pipeline
from grain_test_sql.schema_registry import SchemaRegistry
from grain_test_sql.signature_extractor import extract_signature
from grain_test_sql.template_matcher import SYNTHETIC_FAULT_CHILD, match_tests


DATASETS = {
    "bird": {
        "eval": "artifacts/datasets/bird/eval/bird_eval_set.jsonl",
        "faults": "artifacts/runs/bird/shared/01_gold_fault_validation/false_candidates_gold_fault_detectable.jsonl",
        "results": "artifacts/runs/bird/shared/01_gold_fault_validation/grain_repair_gold_fault_detectable.jsonl",
    },
    "spider": {
        "eval": "artifacts/datasets/spider/eval/spider_eval_set.jsonl",
        "faults": "artifacts/runs/spider/shared/01_gold_fault_validation/false_candidates_gold_fault_detectable.jsonl",
        "results": "artifacts/runs/spider/shared/01_gold_fault_validation/grain_repair_gold_fault_detectable.jsonl",
    },
}

OPERATORS = ("GRT", "CGR", "DET", "EST", "ART", "TST")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build matched safe controls and a decision-blind witness audit bundle."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/runs/controlled_validity",
    )
    parser.add_argument(
        "--sample-per-stratum",
        type=int,
        default=5,
        help="Deterministic witness sample size per dataset/operator stratum; 0 audits all.",
    )
    parser.add_argument(
        "--human-review-tsv",
        default="",
        help="Optional completed blind-review TSV to add human independent-audit columns.",
    )
    parser.add_argument(
        "--dataset-config-json",
        default="",
        help=(
            "Optional JSON object mapping dataset names to {eval, faults, results} "
            "paths relative to the repository root. This permits versioned replay "
            "runs without overwriting the archived controlled-validity inputs."
        ),
    )
    args = parser.parse_args()

    datasets = DATASETS
    if args.dataset_config_json:
        parsed = json.loads(args.dataset_config_json)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("--dataset-config-json must be a non-empty JSON object")
        for dataset, paths in parsed.items():
            if not isinstance(paths, dict) or not {"eval", "faults", "results"} <= set(paths):
                raise ValueError(
                    f"dataset {dataset!r} must provide eval, faults, and results paths"
                )
        datasets = parsed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_rows: list[dict[str, Any]] = []
    safe_results: list[dict[str, Any]] = []
    witness_candidates: list[dict[str, Any]] = []
    faulty_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    detected_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for dataset, paths in datasets.items():
        eval_rows = index_eval_rows(read_jsonl(ROOT / paths["eval"]))
        fault_rows = read_jsonl(ROOT / paths["faults"])
        result_rows = read_jsonl(ROOT / paths["results"])
        faults_by_key = {row_key(row): row for row in fault_rows}

        registries: dict[str, SchemaRegistry] = {}
        for result_row in result_rows:
            operator = result_row["contract_type"]
            faulty_counts[dataset][operator] += 1
            if result_row.get("grain_result", {}).get("status") == "VIOLATION":
                detected_counts[dataset][operator] += 1

            key = row_key(result_row)
            fault_row = faults_by_key[key]
            eval_row = eval_rows[(result_row["question_id"], result_row["db_id"])]
            registry_path = str(ROOT / eval_row["schema_registry_path"])
            registry = registries.setdefault(registry_path, SchemaRegistry.from_yaml(registry_path))
            contract = MetricContract.from_dict(result_row["contract"])

            safe_sql, strategy = build_safe_control(fault_row, result_row, contract)
            safe_result = run_pipeline(safe_sql, registry, contract).to_dict()
            safe_id = stable_id(dataset, result_row, prefix="safe")
            safe_rows.append(
                {
                    "safe_control_id": safe_id,
                    "dataset": dataset,
                    "question_id": result_row["question_id"],
                    "db_id": result_row["db_id"],
                    "contract_type": operator,
                    "contract_index": result_row.get("contract_index", 0),
                    "fault_source": fault_row.get("source"),
                    "faulty_sql": fault_row.get("candidate_sql") or fault_row.get("candidate sql"),
                    "safe_sql": safe_sql,
                    "safe_strategy": strategy,
                    "contract": result_row["contract"],
                }
            )
            safe_results.append(
                {
                    "safe_control_id": safe_id,
                    "dataset": dataset,
                    "question_id": result_row["question_id"],
                    "db_id": result_row["db_id"],
                    "contract_type": operator,
                    "contract_index": result_row.get("contract_index", 0),
                    "safe_strategy": strategy,
                    "status": safe_result["status"],
                    "false_alarm": safe_result["status"] == "VIOLATION",
                    "grain_result": safe_result,
                }
            )

            if result_row.get("grain_result", {}).get("status") == "VIOLATION":
                witness_candidates.extend(
                    reconstruct_witnesses(dataset, result_row, registry_path, registry, contract)
                )

    sampled_witnesses = stratified_sample(witness_candidates, args.sample_per_stratum)
    prechecks = [independent_precheck(row) for row in sampled_witnesses]
    human_reviews = load_human_reviews(Path(args.human_review_tsv)) if args.human_review_tsv else {}

    write_jsonl(output_dir / "safe_controls.jsonl", safe_rows)
    write_jsonl(output_dir / "safe_control_validation.jsonl", safe_results)
    write_jsonl(output_dir / "witness_blind_sample.jsonl", [blind_witness(row) for row in sampled_witnesses])
    write_jsonl(output_dir / "witness_independent_code_audit.jsonl", prechecks)
    write_review_template(output_dir / "witness_blind_review.tsv", sampled_witnesses, human_reviews)

    summary = build_summary(
        faulty_counts,
        detected_counts,
        safe_results,
        prechecks,
        human_reviews,
        datasets=tuple(datasets),
        witness_audit_scope=(
            "full_census" if args.sample_per_stratum == 0 else "stratified_sample"
        ),
    )
    summary["input_datasets"] = datasets
    summary["sample_per_stratum"] = args.sample_per_stratum
    (output_dir / "controlled_validity.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_table(
        output_dir / "controlled_validity.table.tsv",
        [*summary["by_operator"], summary["total"]],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_safe_control(
    fault_row: dict[str, Any],
    result_row: dict[str, Any],
    contract: MetricContract,
) -> tuple[str, str]:
    source = fault_row.get("source")
    if source == "gold_sql_fault_injection":
        return result_row["gold_sql"], "pre_injection_gold_sql"
    if source == "contract_synthetic_negative":
        # This older fallback is synthesized from the contract rather than by
        # editing the reference SQL.  Its paired safe variant is therefore the
        # same-contract reference query, explicitly recorded as such instead
        # of silently excluding the fault from the control audit.
        return result_row["gold_sql"], "reference_gold_sql_control"
    if source != "contract_synthetic_forced":
        raise ValueError(f"unsupported fault source: {source}")

    sql = fault_row.get("candidate_sql") or fault_row.get("candidate sql")
    operator = result_row["contract_type"]
    measure = contract.measures[0]
    tree = sqlglot.parse_one(sql, read="sqlite")

    if operator in {"DET", "CGR", "GRT", "ART"}:
        changed = False
        for if_node in tree.find_all(exp.If):
            if_node.set("true", exp.Literal.number(0))
            changed = True
        if not changed and operator == "GRT":
            count = next(tree.find_all(exp.Count), None)
            if count is not None:
                count.replace(exp.Sum(this=exp.Literal.number(0)))
                changed = True
        if not changed:
            raise ValueError(f"could not neutralize synthetic {operator} control")
        return tree.sql(dialect="sqlite"), "neutralized_synthetic_trigger"

    if operator == "EST":
        if not measure.source_column:
            raise ValueError("EST safe control requires source_column")
        avg = next(tree.find_all(exp.Count), None)
        if avg is None:
            raise ValueError("EST synthetic candidate has no COUNT")
        avg.replace(exp.Sum(this=exp.column(measure.source_column)))
        return tree.sql(dialect="sqlite"), "entity_total_preserving_sum"

    if operator == "TST":
        order = tree.args.get("order")
        if not order:
            raise ValueError("TST synthetic candidate has no ORDER BY")
        select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
        if select is None:
            raise ValueError("TST synthetic candidate has no SELECT")
        metric_projection = next(
            (
                projection
                for projection in select.expressions
                if isinstance(projection, exp.Alias)
                and projection.alias_or_name.lower() == "ranking_metric"
            ),
            None,
        )
        if metric_projection is None:
            raise ValueError("TST synthetic candidate has no ranking_metric projection")
        metric_expression = metric_projection.this.copy()
        for ordered in order.expressions:
            ordered.set("this", metric_expression.copy())
            ordered.set("desc", True)
        select.set(
            "expressions",
            [projection for projection in select.expressions if projection is not metric_projection],
        )
        return tree.sql(dialect="sqlite"), "stable_top_group_with_unprojected_metric"

    raise ValueError(f"unsupported operator: {operator}")


def reconstruct_witnesses(
    dataset: str,
    result_row: dict[str, Any],
    registry_path: str,
    registry: SchemaRegistry,
    contract: MetricContract,
) -> list[dict[str, Any]]:
    sql = result_row.get("candidate sql") or result_row.get("candidate_sql")
    signature = extract_signature(sql)
    builder = PairBuilder(
        registry,
        contract,
        filters=signature.filters,
        conditional_filters=signature.conditional_filters,
        required_tables=signature.tables,
        join_column_pairs=signature.join_column_pairs,
        group_columns=signature.groups,
    )
    archived_violations = {
        (item.get("test_type"), item.get("measure"))
        for item in result_row.get("grain_result", {}).get("results", [])
        if item.get("status") == "VIOLATION"
    }
    rows: list[dict[str, Any]] = []
    for measure in contract.measures:
        for plan_index, plan in enumerate(match_tests(measure, signature, registry)):
            if plan.static_status or (plan.test_type, measure.measure_id) not in archived_violations:
                continue
            pair = builder.build(plan)
            out0 = execute_sql(sql, pair.d0, registry)
            out1 = execute_sql(sql, pair.d1, registry)
            audit_id = stable_id(dataset, result_row, prefix=f"wit-{plan_index}")
            rows.append(
                {
                    "audit_id": audit_id,
                    "dataset": dataset,
                    "operator": plan.test_type,
                    "question_id": result_row["question_id"],
                    "db_id": result_row["db_id"],
                    "contract_index": result_row.get("contract_index", 0),
                    "registry_path": registry_path,
                    "contract": result_row["contract"],
                    "measure_id": measure.measure_id,
                    "plan": {
                        "test_type": plan.test_type,
                        "child_table": plan.child_table,
                    },
                    "pair_meta": pair.meta,
                    "d0": pair.d0,
                    "d1": pair.d1,
                    "replayed_output_d0": out0,
                    "replayed_output_d1": out1,
                }
            )
            # The controlled-validity table counts retained witnesses, one per
            # detected SQL input, rather than every dynamic plan that happens
            # to expose the same injected fault.
            return rows
    if not rows:
        raise ValueError(
            f"could not reconstruct archived violation {dataset}/{result_row['question_id']}/{result_row['contract_type']}"
        )
    return rows


def stratified_sample(rows: list[dict[str, Any]], per_stratum: int) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(row["dataset"], row["operator"])].append(row)
    sampled: list[dict[str, Any]] = []
    for key in sorted(strata):
        ordered = sorted(strata[key], key=lambda row: row["audit_id"])
        sampled.extend(ordered if per_stratum == 0 else ordered[:per_stratum])
    return sampled


def independent_precheck(row: dict[str, Any]) -> dict[str, Any]:
    registry = SchemaRegistry.from_yaml(row["registry_path"])
    schema_errors = schema_errors_for_pair(row["d0"], row["d1"], registry)
    c0 = independent_contract_value(row, row["d0"])
    c1 = independent_contract_value(row, row["d1"])
    perturbation_ok, perturbation_reason = perturbation_isolated(row, registry)
    replay_changed = normalized_json(row["replayed_output_d0"]) != normalized_json(row["replayed_output_d1"])
    valid = not schema_errors and c0 == c1 and perturbation_ok and replay_changed
    return {
        "audit_id": row["audit_id"],
        "dataset": row["dataset"],
        "operator": row["operator"],
        "schema_valid": not schema_errors,
        "schema_errors": schema_errors,
        "contract_value_d0": c0,
        "contract_value_d1": c1,
        "contract_preserved": c0 == c1,
        "perturbation_isolated": perturbation_ok,
        "perturbation_reason": perturbation_reason,
        "violation_replayed": replay_changed,
        "code_audit_valid": valid,
        "audit_scope": "decision-blind independent code precheck; not a human independent review",
    }


def schema_errors_for_pair(
    d0: dict[str, list[dict[str, Any]]],
    d1: dict[str, list[dict[str, Any]]],
    registry: SchemaRegistry,
) -> list[str]:
    errors: list[str] = []
    for label, database in (("D0", d0), ("D1", d1)):
        for table_name, rows in database.items():
            if table_name not in registry.tables:
                errors.append(f"{label}: unknown table {table_name}")
                continue
            schema = registry.tables[table_name]
            for index, row in enumerate(rows):
                unknown = set(row) - set(schema.columns)
                if unknown:
                    errors.append(f"{label}:{table_name}[{index}] unknown columns {sorted(unknown)}")
            for columns in [schema.primary_key, *schema.unique]:
                if not columns:
                    continue
                values = [tuple(item.get(column) for column in columns) for item in rows]
                non_null = [value for value in values if all(part is not None for part in value)]
                if len(non_null) != len(set(non_null)):
                    errors.append(f"{label}:{table_name} duplicate unique key {columns}")
            for fk in schema.foreign_keys:
                parent_rows = database.get(fk.ref_table, [])
                parent_keys = {
                    tuple(parent.get(column) for column in fk.ref_columns)
                    for parent in parent_rows
                }
                for index, child in enumerate(rows):
                    key = tuple(child.get(column) for column in fk.columns)
                    if any(part is None for part in key):
                        continue
                    if key not in parent_keys:
                        errors.append(
                            f"{label}:{table_name}[{index}] FK {fk.columns}->{fk.ref_table}.{fk.ref_columns} missing {key}"
                        )
    return sorted(set(errors))


def independent_contract_value(row: dict[str, Any], database: dict[str, list[dict[str, Any]]]) -> Any:
    contract = row["contract"]
    measure = next(item for item in contract["measures"] if item["measure_id"] == row["measure_id"])
    formula = measure["semantic_formula"]
    source = measure["source_table"]
    source_rows = database[source]

    if formula in {"SUM_ONCE_PER_GRAIN", "AVG_ONCE_PER_GRAIN"}:
        columns = [ref.split(".", 1)[1] for ref in measure["base_grain"]]
        unique_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        for item in source_rows:
            unique_rows.setdefault(tuple(item[column] for column in columns), item)
        values = [item[measure["source_column"]] for item in unique_rows.values()]
        return sum(values) if formula.startswith("SUM") else safe_mean(values)
    if formula == "COUNT_ONCE_PER_GRAIN":
        columns = [ref.split(".", 1)[1] for ref in measure["base_grain"]]
        return len({tuple(item[column] for column in columns) for item in source_rows})
    if formula == "COUNT_DISTINCT_ENTITY":
        table, column = measure["entity_key"].split(".", 1)
        if table in database and database[table]:
            return len({item[column] for item in database[table]})
        event_cols = row.get("pair_meta", {}).get("event_entity_cols") or []
        events = database[measure.get("event_table") or source]
        return len({tuple(item[col] for col in event_cols) for item in events})
    if formula == "AVG_ENTITY_TOTAL":
        event_cols = row.get("pair_meta", {}).get("event_entity_cols") or []
        _, entity_col = measure["entity_key"].split(".", 1)
        totals: dict[Any, float] = defaultdict(float)
        for item in source_rows:
            key = tuple(item[col] for col in event_cols) if event_cols else item[entity_col]
            totals[key] += item[measure["source_column"]]
        return safe_mean(list(totals.values()))
    if formula == "AVG_ONCE_PER_ENTITY":
        return safe_mean([item[measure["source_column"]] for item in source_rows])
    if formula == "TOPK_BY_METRIC":
        return row.get("pair_meta", {}).get("expected_top_group", "East")
    raise ValueError(f"unsupported formula in independent audit: {formula}")


def perturbation_isolated(
    row: dict[str, Any],
    registry: SchemaRegistry,
) -> tuple[bool, str]:
    changed_tables = {
        table
        for table in set(row["d0"]) | set(row["d1"])
        if normalized_json(row["d0"].get(table, [])) != normalized_json(row["d1"].get(table, []))
    }
    if not changed_tables:
        return False, "D0 and D1 are identical"

    operator = row["operator"]
    measure = next(
        item for item in row["contract"]["measures"] if item["measure_id"] == row["measure_id"]
    )
    child = row.get("plan", {}).get("child_table")
    expected = {measure["source_table"]}
    if child and child != SYNTHETIC_FAULT_CHILD:
        expected.add(child)
    if measure.get("event_table"):
        expected.add(measure["event_table"])
    if measure.get("entity_key") and "." in measure["entity_key"]:
        expected.add(measure["entity_key"].split(".", 1)[0])
    output_group = row["contract"].get("output_group", {}).get("table")
    if operator == "TST" and output_group:
        expected.add(output_group)

    # Constraint completion may add or vary FK ancestors of an intended
    # perturbation table. These rows are admissible schema auxiliaries, not a
    # second semantic perturbation.
    changed = True
    while changed:
        changed = False
        for table in list(expected):
            if table not in registry.tables:
                continue
            for fk in registry.tables[table].foreign_keys:
                if fk.ref_table not in expected:
                    expected.add(fk.ref_table)
                    changed = True

    unexpected = changed_tables - expected
    if unexpected:
        return False, f"unexpected changed tables: {sorted(unexpected)}"
    return True, f"changes confined to operator-relevant tables: {sorted(changed_tables)}"


def blind_witness(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "audit_id": row["audit_id"],
        "dataset": row["dataset"],
        "operator": row["operator"],
        "registry_path": row["registry_path"],
        "contract": row["contract"],
        "measure_id": row["measure_id"],
        "plan": row["plan"],
        "pair_meta": row["pair_meta"],
        "d0": row["d0"],
        "d1": row["d1"],
        "review_note": "Verifier decision and SQL outputs intentionally omitted.",
    }


def build_summary(
    faulty_counts: dict[str, dict[str, int]],
    detected_counts: dict[str, dict[str, int]],
    safe_results: list[dict[str, Any]],
    prechecks: list[dict[str, Any]],
    human_reviews: dict[str, dict[str, Any]],
    *,
    datasets: tuple[str, ...],
    witness_audit_scope: str,
) -> dict[str, Any]:
    safe_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit_by_operator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in safe_results:
        safe_by_operator[row["contract_type"]].append(row)
    for row in prechecks:
        audit_by_operator[row["operator"]].append(row)

    operator_rows: list[dict[str, Any]] = []
    for operator in OPERATORS:
        safe = safe_by_operator[operator]
        audits = audit_by_operator[operator]
        reviewed = [human_reviews[item["audit_id"]] for item in audits if item["audit_id"] in human_reviews]
        operator_rows.append(
            {
                "operator": operator,
                "faulty_variants": sum(faulty_counts[dataset][operator] for dataset in datasets),
                "detected": sum(detected_counts[dataset][operator] for dataset in datasets),
                "safe_variants": len(safe),
                "false_alarms": sum(bool(item["false_alarm"]) for item in safe),
                "safe_pass": sum(item["status"] == "PASS" for item in safe),
                "safe_unsupported": sum(item["status"] not in {"PASS", "VIOLATION"} for item in safe),
                "code_audited_witnesses": len(audits),
                "code_valid_witnesses": sum(bool(item["code_audit_valid"]) for item in audits),
                "human_audited_witnesses": len(reviewed),
                "human_valid_witnesses": sum(bool(item.get("valid_witness")) for item in reviewed),
            }
        )
    total = {
        key: sum(int(row[key]) for row in operator_rows)
        for key in operator_rows[0]
        if key != "operator"
    }
    total["operator"] = "Total"
    return {
        "audit_protocol": {
            "safe_control": "pre-injection GOLD for structural and contract-synthetic-negative mutations; trigger-neutralized matched SQL for forced synthetic mutations",
            "code_audit": (
                f"{witness_audit_scope}, verifier decision omitted; independently "
                "checks schema/contract/perturbation/replay"
            ),
            "human_audit": "populated only from a completed --human-review-tsv; code audit is not reported as human independent review",
        },
        "by_operator": operator_rows,
        "total": total,
    }


def write_review_template(
    path: Path,
    sampled: list[dict[str, Any]],
    reviews: dict[str, dict[str, Any]],
) -> None:
    fields = [
        "audit_id",
        "dataset",
        "operator",
        "schema_valid",
        "contract_preserved",
        "perturbation_isolated",
        "valid_witness",
        "reviewer",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sampled:
            prior = reviews.get(row["audit_id"], {})
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "dataset": row["dataset"],
                    "operator": row["operator"],
                    **{field: prior.get(field, "") for field in fields[3:]},
                }
            )


def load_human_reviews(path: Path) -> dict[str, dict[str, Any]]:
    reviews: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not row.get("valid_witness", "").strip():
                continue
            converted = dict(row)
            for field in ("schema_valid", "contract_preserved", "perturbation_isolated", "valid_witness"):
                converted[field] = parse_bool(row.get(field, ""))
            reviews[row["audit_id"]] = converted
    return reviews


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def index_eval_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, Any], dict[str, Any]]:
    return {(row["question_id"], row["db_id"]): row for row in rows}


def row_key(row: dict[str, Any]) -> tuple[Any, Any, str, int]:
    return (
        row["question_id"],
        row["db_id"],
        row["contract_type"],
        int(row.get("contract_index", 0)),
    )


def stable_id(dataset: str, row: dict[str, Any], prefix: str) -> str:
    payload = "|".join(
        [
            dataset,
            str(row["question_id"]),
            row["db_id"],
            row["contract_type"],
            str(row.get("contract_index", 0)),
            prefix,
        ]
    )
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def normalized_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


if __name__ == "__main__":
    main()
