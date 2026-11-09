#!/usr/bin/env python3
"""Run deterministic evidence-budget sensitivity on controlled faulty SQL.

The current validator constructs one canonical pair for each dynamic test plan.
For this sensitivity study, that pair is instantiated with a fixed ordered set
of ten numeric value profiles.  The schema, contract, filters, plan order, and
row order remain fixed.  ``all`` means all profiles, not an unbounded search.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from grain_test_sql.candidate_builder import build_candidate_space
from grain_test_sql.contract import MetricContract
from grain_test_sql.contract_checker import preserves_contract
from grain_test_sql.contract_gate import validate_contract
from grain_test_sql.evidence_scope import SUPPORTED, scoped_test_plans
from grain_test_sql.pair_builder import PairBuilder, TestPair
from grain_test_sql.schema_registry import SchemaRegistry
from grain_test_sql.signature_extractor import FilterSignature, extract_signature
from grain_test_sql.template_matcher import TestPlan
from grain_test_sql.validator import validate_pair


DEFAULT_INPUTS = (
    (
        "bird",
        ROOT
        / "code/artifacts/runs/bird/shared/01_gold_fault_validation/"
        "grain_repair_gold_fault_detectable.jsonl",
    ),
    (
        "spider",
        ROOT
        / "code/artifacts/runs/spider/shared/01_gold_fault_validation/"
        "grain_repair_gold_fault_detectable.jsonl",
    ),
)
DEFAULT_REGISTRIES = {
    "bird": ROOT / "code/artifacts/datasets/bird/eval/registries",
    "spider": ROOT / "code/artifacts/datasets/spider/eval/registries",
}
DEFAULT_OUTPUT_DIR = ROOT / "code/artifacts/runs/budget_sensitivity"
BUDGETS: tuple[int | None, ...] = (1, 3, 6, 10, None)
# (multiplier, offset) is applied to every numeric cell not fixed by a SQL
# filter. Affine transforms preserve key equality and D0/D1 contract equality.
VALUE_PROFILES: tuple[tuple[int, int], ...] = (
    (1, 0),
    (2, 0),
    (3, 0),
    (1, 9),
    (2, 9),
    (4, 0),
    (1, 19),
    (3, 7),
    (5, 0),
    (2, 19),
)


@dataclass
class PlanTrace:
    key: str
    candidate_count: int
    execution_statuses: list[str]
    execution_seconds: list[float]
    first_violation_rank: int | None
    build_error: str | None = None
    evidence_scope: str = SUPPORTED
    scope_reason: str | None = None


@dataclass
class RowTrace:
    dataset: str
    sample_id: str
    plan_traces: list[PlanTrace]
    static_violation: bool = False
    static_pass: bool = False
    static_guard_status: str | None = None
    unsupported_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="DATASET=JSONL",
        help="Repeatable input override; defaults to BIRD and Spider controlled faults.",
    )
    parser.add_argument(
        "--registry-dir",
        action="append",
        default=[],
        metavar="DATASET=DIR",
        help="Repeatable schema-registry directory override.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Limit total rows for a smoke test.")
    parser.add_argument(
        "--budgets",
        default="1,3,6,10,all",
        help="Comma-separated positive integers and/or all.",
    )
    return parser.parse_args()


def parse_mapping(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected DATASET=PATH, got {value!r}")
        dataset, raw_path = value.split("=", 1)
        result[dataset.strip()] = Path(raw_path).expanduser().resolve()
    return result


def parse_budgets(value: str) -> tuple[int | None, ...]:
    parsed: list[int | None] = []
    for token in value.split(","):
        token = token.strip().casefold()
        if token == "all":
            parsed.append(None)
            continue
        budget = int(token)
        if budget <= 0:
            raise ValueError("budgets must be positive")
        parsed.append(budget)
    if None not in parsed:
        raise ValueError("budgets must include all as the exhaustive reference")
    return tuple(parsed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def protected_filter_columns(
    filters: list[FilterSignature], conditional: list[FilterSignature]
) -> set[tuple[str, str]]:
    protected: set[tuple[str, str]] = set()
    for item in [*filters, *conditional]:
        if "." in item.source:
            table, column = item.source.split(".", 1)
            protected.add((table.casefold(), column.casefold()))
    return protected


def transform_rows(
    rows: dict[str, list[dict[str, Any]]],
    multiplier: int,
    offset: int,
    protected: set[tuple[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    transformed = copy.deepcopy(rows)
    for table, table_rows in transformed.items():
        for row in table_rows:
            for column, value in row.items():
                if (table.casefold(), column.casefold()) in protected:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                row[column] = value * multiplier + offset
    return transformed


def ordered_pairs(
    builder: PairBuilder,
    plan: TestPlan,
    filters: list[FilterSignature],
    conditional: list[FilterSignature],
) -> list[TestPair]:
    base = builder.build(plan)
    protected = protected_filter_columns(filters, conditional)
    pairs: list[TestPair] = []
    seen: set[str] = set()
    for index, (multiplier, offset) in enumerate(VALUE_PROFILES, start=1):
        pair = TestPair(
            test_type=base.test_type,
            measure=base.measure,
            d0=transform_rows(base.d0, multiplier, offset, protected),
            d1=transform_rows(base.d1, multiplier, offset, protected),
            group_value=base.group_value,
            meta={
                **base.meta,
                "evidence_rank": index,
                "value_profile": {"multiplier": multiplier, "offset": offset},
            },
        )
        fingerprint = json.dumps(
            {"d0": pair.d0, "d1": pair.d1},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if fingerprint not in seen:
            seen.add(fingerprint)
            pairs.append(pair)
    return pairs


def trace_row(
    dataset: str,
    row: dict[str, Any],
    registry: SchemaRegistry,
    *,
    guard_static_pass_violations: bool = False,
    enable_cgr_group_ranking_contrast: bool = True,
    output_comparator: str = "exact",
) -> RowTrace:
    """Trace matched obligations without globally suppressing violations.

    ``guard_static_pass_violations`` remains accepted so archived callers can
    be loaded, but it is intentionally decision-inert. Static safety is
    represented by obligation-local ``plan.static_status`` only.
    """
    sample_id = (
        f"{dataset}:{row.get('db_id')}:{row.get('question_id')}:"
        f"{row.get('contract_type')}:{row.get('contract_index', 0)}"
    )
    sql = row.get("candidate sql") or row.get("candidate_sql") or ""
    contract_obj = row.get("contract")
    if not sql or not contract_obj:
        return RowTrace(dataset, sample_id, [], unsupported_reason="missing SQL or contract")
    try:
        contract = MetricContract.from_dict(contract_obj)
        gate = validate_contract(contract, registry, build_candidate_space(registry))
        if not gate.ok:
            return RowTrace(
                dataset, sample_id, [],
                unsupported_reason="unsupported contract: " + "; ".join(gate.errors),
            )
        signature = extract_signature(sql)
        static_guard_status = None
    except Exception as exc:
        return RowTrace(
            dataset, sample_id, [],
            unsupported_reason=f"setup error: {type(exc).__name__}: {exc}",
        )

    builder = PairBuilder(
        registry,
        contract,
        filters=signature.filters,
        conditional_filters=signature.conditional_filters,
        required_tables=signature.tables,
        join_column_pairs=signature.join_column_pairs,
        group_columns=signature.groups,
        scalar_aggregate_comparisons=signature.scalar_aggregate_comparisons,
        projection_columns=signature.direct_projections,
        ranking_selection=signature.has_order_limit,
        ranking_count_sources=[
            measure.source
            for measure in signature.measures
            if measure.alias == "ranking_metric"
            and measure.function == "COUNT"
        ],
        enable_cgr_group_ranking_contrast=enable_cgr_group_ranking_contrast,
    )
    traces: list[PlanTrace] = []
    static_violation = False
    static_pass = False
    for measure in contract.measures:
        for plan_index, scoped_plan in enumerate(
            scoped_test_plans(
                measure,
                signature,
                registry,
                output_group_ref=contract.output_group.ref,
            )
        ):
            plan = scoped_plan.plan
            plan_key = f"{measure.measure_id}:{plan.test_type}:{plan.child_table}:{plan_index}"
            if plan.static_status:
                static_violation = static_violation or plan.static_status == "VIOLATION"
                static_pass = static_pass or plan.static_status == "PASS"
                continue
            try:
                pairs = ordered_pairs(
                    builder, plan, signature.filters, signature.conditional_filters
                )
            except Exception as exc:
                traces.append(
                    PlanTrace(
                        key=plan_key,
                        candidate_count=0,
                        execution_statuses=[],
                        execution_seconds=[],
                        first_violation_rank=None,
                        build_error=f"{type(exc).__name__}: {exc}",
                        evidence_scope=scoped_plan.evidence_scope,
                        scope_reason=scoped_plan.scope_reason,
                    )
                )
                continue
            statuses: list[str] = []
            seconds: list[float] = []
            first_violation: int | None = None
            for rank, pair in enumerate(pairs, start=1):
                started = time.perf_counter()
                try:
                    if not preserves_contract(pair):
                        status = "INVALID_WITNESS"
                    else:
                        status = validate_pair(
                            sql,
                            pair,
                            registry,
                            output_comparator=output_comparator,
                        ).status
                except Exception:
                    status = "INVALID_WITNESS"
                seconds.append(time.perf_counter() - started)
                statuses.append(status)
                if status == "VIOLATION" and first_violation is None:
                    first_violation = rank
            traces.append(
                PlanTrace(
                    key=plan_key,
                    candidate_count=len(pairs),
                    execution_statuses=statuses,
                    execution_seconds=seconds,
                    first_violation_rank=first_violation,
                    evidence_scope=scoped_plan.evidence_scope,
                    scope_reason=scoped_plan.scope_reason,
                )
            )
    return RowTrace(
        dataset,
        sample_id,
        traces,
        static_violation=static_violation,
        static_pass=static_pass,
        static_guard_status=static_guard_status,
    )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_budget(traces: list[RowTrace], budget: int | None) -> dict[str, Any]:
    detected_rows = 0
    retained_plans = 0
    unsupported_rows = 0
    exhausted_rows = 0
    executed = 0
    plan_count = 0
    row_seconds: list[float] = []
    status_counts: Counter[str] = Counter()
    for row in traces:
        row_detected = row.static_violation
        row_incomplete = False
        row_has_valid_plan = False
        elapsed = 0.0
        if row.unsupported_reason:
            unsupported_rows += 1
            status_counts["UNSUPPORTED"] += 1
            row_seconds.append(0.0)
            continue
        for plan in row.plan_traces:
            plan_count += 1
            if plan.candidate_count == 0:
                continue
            row_has_valid_plan = True
            cap = plan.candidate_count if budget is None else min(
                budget, plan.candidate_count
            )
            violation_rank = plan.first_violation_rank
            used = cap
            if violation_rank is not None and violation_rank <= cap:
                used = violation_rank
                retained_plans += 1
                row_detected = True
            elif cap < plan.candidate_count:
                row_incomplete = True
            executed += used
            elapsed += sum(plan.execution_seconds[:used])
        row_seconds.append(elapsed)
        if row_detected:
            detected_rows += 1
            status_counts["VIOLATION"] += 1
        elif row_incomplete:
            exhausted_rows += 1
            unsupported_rows += 1
            status_counts["EVIDENCE_BUDGET_EXHAUSTED"] += 1
        elif not row_has_valid_plan:
            if row.static_pass:
                status_counts["PASS"] += 1
            else:
                unsupported_rows += 1
                status_counts["NO_VALID_EVIDENCE"] += 1
        else:
            status_counts["PASS"] += 1
    return {
        "budget": "all" if budget is None else budget,
        "rows": len(traces),
        "detected_faulty_rows": detected_rows,
        "retained_witnesses": detected_rows,
        "retained_witness_plans": retained_plans,
        "unsupported_rows": unsupported_rows,
        "unsupported_rate": unsupported_rows / len(traces) if traces else None,
        "budget_exhausted_rows": exhausted_rows,
        "dynamic_plans": plan_count,
        "executions": executed,
        "mean_executions_per_plan": executed / plan_count if plan_count else None,
        "median_runtime_ms_per_row": (
            statistics.median(row_seconds) * 1000 if row_seconds else None
        ),
        "p95_runtime_ms_per_row": (
            percentile(row_seconds, 0.95) * 1000 if row_seconds else None
        ),
        "status_counts": dict(sorted(status_counts.items())),
    }


def write_paper_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Write the five columns reported in the paper's budget table."""
    columns = (
        "budget",
        "retained_witnesses",
        "retention_vs_all_pct",
        "unsupported_pct",
        "mean_executions_per_plan",
    )
    lines = ["\t".join(columns)]
    for item in summaries:
        retention = item["retention_vs_all"]
        unsupported = item["unsupported_rate"]
        mean_executions = item["mean_executions_per_plan"]
        lines.append(
            "\t".join(
                (
                    str(item["budget"]),
                    str(item["retained_witnesses"]),
                    "" if retention is None else f"{retention * 100:.2f}",
                    "" if unsupported is None else f"{unsupported * 100:.2f}",
                    "" if mean_executions is None else f"{mean_executions:.4f}",
                )
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    input_overrides = parse_mapping(args.input)
    registry_overrides = parse_mapping(args.registry_dir)
    inputs = dict(DEFAULT_INPUTS)
    if input_overrides:
        inputs = input_overrides
    registries = {**DEFAULT_REGISTRIES, **registry_overrides}

    work: list[tuple[str, dict[str, Any]]] = []
    input_manifest: dict[str, Any] = {}
    for dataset, path in inputs.items():
        rows = read_jsonl(path)
        input_manifest[dataset] = {"path": str(path), "rows": len(rows)}
        work.extend((dataset, row) for row in rows)
    if args.limit:
        work = work[: args.limit]

    registry_cache: dict[tuple[str, str], SchemaRegistry] = {}
    traces: list[RowTrace] = []
    for index, (dataset, row) in enumerate(work, start=1):
        db_id = str(row.get("db_id") or "")
        registry_path = registries[dataset] / f"{db_id}.yaml"
        cache_key = (dataset, db_id)
        if cache_key not in registry_cache:
            registry_cache[cache_key] = SchemaRegistry.from_yaml(registry_path)
        traces.append(trace_row(dataset, row, registry_cache[cache_key]))
        if index % 50 == 0 or index == len(work):
            print(f"traced {index}/{len(work)} rows", flush=True)

    summaries = [summarize_budget(traces, budget) for budget in budgets]
    exhaustive = next(item for item in summaries if item["budget"] == "all")
    exhaustive_witnesses = exhaustive["retained_witnesses"]
    for item in summaries:
        item["retention_vs_all"] = (
            item["retained_witnesses"] / exhaustive_witnesses
            if exhaustive_witnesses else None
        )
    report = {
        "experiment": "evidence_budget_sensitivity_v1",
        "definition": {
            "budget_unit": "numeric value-profile executions per dynamic plan",
            "ordered_value_profiles": [
                {"multiplier": multiplier, "offset": offset}
                for multiplier, offset in VALUE_PROFILES
            ],
            "all_definition": "all ten ordered profiles (or fewer after exact deduplication)",
            "retained_witness_definition": (
                "faulty SQL rows with at least one dynamic plan producing a "
                "VIOLATION within the budget prefix"
            ),
            "retained_witness_plans_definition": (
                "dynamic plans with a VIOLATION within the budget prefix; "
                "reported separately because a row can have multiple plans"
            ),
            "unsupported_definition": (
                "setup/no-valid-evidence rows plus undetected rows whose ordered "
                "evidence list is unfinished at the budget cap"
            ),
            "early_stop": "stop a plan at its first VIOLATION",
        },
        "inputs": input_manifest,
        "evaluated_rows": len(traces),
        "budgets": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "budget_sensitivity.summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table_path = args.output_dir / "budget_sensitivity.table.tsv"
    write_paper_table(table_path, summaries)
    trace_path = args.output_dir / "budget_sensitivity.traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in traces:
            handle.write(
                json.dumps(
                    {
                        "dataset": row.dataset,
                        "sample_id": row.sample_id,
                        "static_violation": row.static_violation,
                        "static_pass": row.static_pass,
                        "static_guard_status": row.static_guard_status,
                        "unsupported_reason": row.unsupported_reason,
                        "plans": [
                            {
                                "key": plan.key,
                                "candidate_count": plan.candidate_count,
                                "execution_statuses": plan.execution_statuses,
                                "execution_seconds": plan.execution_seconds,
                                "first_violation_rank": plan.first_violation_rank,
                                "build_error": plan.build_error,
                                "evidence_scope": plan.evidence_scope,
                                "scope_reason": plan.scope_reason,
                            }
                            for plan in row.plan_traces
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"paper table: {table_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
