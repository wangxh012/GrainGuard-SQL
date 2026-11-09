#!/usr/bin/env python3
"""Run evidence-budget sensitivity on independently labelled natural candidates.

The evidence search is identical to ``run_budget_sensitivity.py``.  The four
frozen model-generated candidate pools are traced without consulting their
labels; labels are used only afterwards to compute effectiveness and
exhaustive-relative witness retention.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_budget_sensitivity import (  # noqa: E402
    DEFAULT_REGISTRIES,
    VALUE_PROFILES,
    RowTrace,
    SchemaRegistry,
    parse_budgets,
    percentile,
    read_jsonl,
    trace_row,
)
from grain_test_sql.bird_utils import (  # noqa: E402
    registry_with_filter_uniqueness_profile,
)
from grain_test_sql.evidence_scope import (  # noqa: E402
    EVIDENCE_SCOPE_POLICIES,
    SUPPORTED_POLICY,
    witness_is_admissible,
)


DEFAULT_INPUTS = (
    (
        "bird_glm",
        "bird",
        ROOT
        / "code/artifacts/runs/final_metrics/bird_glm_candidate.jsonl",
    ),
    (
        "bird_qwen",
        "bird",
        ROOT
        / "code/artifacts/runs/final_metrics/bird_qwen_candidate.jsonl",
    ),
    (
        "spider_glm",
        "spider",
        ROOT
        / "code/artifacts/runs/ablations/final_candidate_inputs/"
        "spider_glm_candidate.jsonl",
    ),
    (
        "spider_qwen",
        "spider",
        ROOT
        / "code/artifacts/runs/ablations/final_candidate_inputs/"
        "spider_qwen_candidate.jsonl",
    ),
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "code/artifacts/runs/natural_candidate_budget_sensitivity"
)
FINAL_LABELS = (
    ROOT
    / "code/artifacts/annotations/sql_labels/"
    "final_reference_labels.paper_core.v2.jsonl"
)
BIRD_DATABASES = (
    ROOT / "code/artifacts/datasets/bird/raw/dev_20240627/dev_databases"
)


@dataclass
class NaturalTrace:
    pool: str
    dataset: str
    label: bool
    trace: RowTrace


@dataclass
class BudgetOutcome:
    status: str
    detected: bool
    executions: int
    plan_count: int
    elapsed_seconds: float
    raw_detected: bool = False
    deferred_witness: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--budgets",
        default="1,3,6,10,all",
        help="Comma-separated positive integers and/or all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the combined input for a smoke test.",
    )
    parser.add_argument(
        "--evidence-scope-policy",
        choices=EVIDENCE_SCOPE_POLICIES,
        default=SUPPORTED_POLICY,
        help="Witness-admissibility policy fixed before evidence execution.",
    )
    return parser.parse_args()


def label_key(dataset: str, pool: str, question_id: Any) -> str:
    model = "glm-5.1" if pool.endswith("_glm") else "qwen3.6-27b"
    return f"{dataset}:candidate_sql:{model}:{question_id}"


def read_final_labels() -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(FINAL_LABELS):
        if row.get("sql_role") != "candidate_sql":
            continue
        labels[str(row["sample_id"])] = row
    return labels


def compact_sql(sql: Any) -> str:
    return re.sub(r"\s+", "", str(sql or ""))


def classify_at_budget(
    row: RowTrace,
    budget: int | None,
    *,
    evidence_scope_policy: str = SUPPORTED_POLICY,
) -> BudgetOutcome:
    if row.unsupported_reason:
        return BudgetOutcome("UNSUPPORTED", False, 0, 0, 0.0)

    detected = row.static_violation
    raw_detected = row.static_violation
    deferred_witness = False
    incomplete = False
    has_dynamic_plan = False
    has_valid_execution = False
    invalid_completed_plan = False
    executions = 0
    elapsed = 0.0
    plan_count = 0

    for plan in row.plan_traces:
        plan_count += 1
        has_dynamic_plan = True
        if plan.candidate_count == 0:
            invalid_completed_plan = True
            continue
        cap = (
            plan.candidate_count
            if budget is None
            else min(budget, plan.candidate_count)
        )
        used = cap
        attempted_statuses = plan.execution_statuses[:cap]
        plan_has_valid_execution = any(
            status in {"PASS", "VIOLATION"}
            for status in attempted_statuses
        )
        has_valid_execution = has_valid_execution or plan_has_valid_execution
        if plan.first_violation_rank is not None and plan.first_violation_rank <= cap:
            used = plan.first_violation_rank
            raw_detected = True
            if witness_is_admissible(plan.evidence_scope, evidence_scope_policy):
                detected = True
            else:
                deferred_witness = True
        elif cap < plan.candidate_count:
            incomplete = True
        elif not plan_has_valid_execution:
            invalid_completed_plan = True
        executions += used
        elapsed += sum(plan.execution_seconds[:used])

    if detected:
        status = "VIOLATION"
    elif deferred_witness:
        status = "ABSTAIN"
    elif incomplete:
        status = "EVIDENCE_BUDGET_EXHAUSTED"
    elif invalid_completed_plan:
        status = "INVALID_WITNESS"
    elif not has_dynamic_plan:
        status = "PASS" if row.static_pass else "NO_VALID_EVIDENCE"
    elif not has_valid_execution:
        status = "INVALID_WITNESS"
    else:
        status = "PASS"
    return BudgetOutcome(
        status,
        detected,
        executions,
        plan_count,
        elapsed,
        raw_detected=raw_detected,
        deferred_witness=deferred_witness,
    )


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def cohort_names(item: NaturalTrace) -> tuple[str, ...]:
    return ("overall", item.dataset, item.pool)


def summarize_cohort(
    rows: list[NaturalTrace],
    budget: int | None,
    exhaustive_detectable_positive_ids: set[str],
    *,
    evidence_scope_policy: str = SUPPORTED_POLICY,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    executions = plan_count = 0
    statuses: Counter[str] = Counter()
    row_seconds: list[float] = []
    detected_positive_ids: set[str] = set()

    for item in rows:
        outcome = classify_at_budget(
            item.trace,
            budget,
            evidence_scope_policy=evidence_scope_policy,
        )
        statuses[outcome.status] += 1
        executions += outcome.executions
        plan_count += outcome.plan_count
        row_seconds.append(outcome.elapsed_seconds)
        if item.label and outcome.detected:
            tp += 1
            detected_positive_ids.add(f"{item.pool}:{item.trace.sample_id}")
        elif item.label:
            fn += 1
        elif outcome.detected:
            fp += 1
        else:
            tn += 1

    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    unsupported = (
        statuses["UNSUPPORTED"]
        + statuses["ABSTAIN"]
        + statuses["EVIDENCE_BUDGET_EXHAUSTED"]
        + statuses["NO_VALID_EVIDENCE"]
        + statuses["INVALID_WITNESS"]
    )
    decision_coverage = safe_ratio(len(rows) - unsupported, len(rows))
    retained_exhaustive_positives = len(
        detected_positive_ids & exhaustive_detectable_positive_ids
    )
    return {
        "budget": "all" if budget is None else budget,
        "rows": len(rows),
        "positives": tp + fn,
        "negatives": fp + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "decision_coverage": decision_coverage,
        "unsupported_rows": unsupported,
        "unsupported_rate": safe_ratio(unsupported, len(rows)),
        "exhaustive_detectable_positives": len(exhaustive_detectable_positive_ids),
        "retained_exhaustive_positives": retained_exhaustive_positives,
        "positive_retention_vs_all": safe_ratio(
            retained_exhaustive_positives,
            len(exhaustive_detectable_positive_ids),
        ),
        "executions": executions,
        "dynamic_plans": plan_count,
        "mean_executions_per_plan": safe_ratio(executions, plan_count),
        "median_runtime_ms_per_row": (
            statistics.median(row_seconds) * 1000 if row_seconds else None
        ),
        "p95_runtime_ms_per_row": (
            percentile(row_seconds, 0.95) * 1000 if row_seconds else None
        ),
        "status_counts": dict(sorted(statuses.items())),
    }


def metric_pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}"


def write_table(
    path: Path,
    cohort_summaries: dict[str, list[dict[str, Any]]],
) -> None:
    columns = (
        "cohort",
        "budget",
        "rows",
        "positives",
        "negatives",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision_pct",
        "recall_pct",
        "f1_pct",
        "decision_coverage_pct",
        "exhaustive_detectable_positives",
        "retained_exhaustive_positives",
        "positive_retention_vs_all_pct",
        "unsupported_rows",
        "unsupported_pct",
        "mean_executions_per_plan",
    )
    lines = ["\t".join(columns)]
    for cohort, summaries in cohort_summaries.items():
        for item in summaries:
            mean_executions = item["mean_executions_per_plan"]
            lines.append(
                "\t".join(
                    (
                        cohort,
                        str(item["budget"]),
                        str(item["rows"]),
                        str(item["positives"]),
                        str(item["negatives"]),
                        str(item["tp"]),
                        str(item["fp"]),
                        str(item["fn"]),
                        str(item["tn"]),
                        metric_pct(item["precision"]),
                        metric_pct(item["recall"]),
                        metric_pct(item["f1"]),
                        metric_pct(item["decision_coverage"]),
                        str(item["exhaustive_detectable_positives"]),
                        str(item["retained_exhaustive_positives"]),
                        metric_pct(item["positive_retention_vs_all"]),
                        str(item["unsupported_rows"]),
                        metric_pct(item["unsupported_rate"]),
                        (
                            ""
                            if mean_executions is None
                            else f"{mean_executions:.4f}"
                        ),
                    )
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    budgets = parse_budgets(args.budgets)
    final_labels = read_final_labels()
    work: list[tuple[str, str, dict[str, Any]]] = []
    input_manifest: dict[str, Any] = {}
    for pool, dataset, path in DEFAULT_INPUTS:
        rows = read_jsonl(path)
        matched_labels = [
            final_labels[label_key(dataset, pool, row.get("question_id"))]
            for row in rows
        ]
        positive_count = sum(
            bool(label["final_paper_core_positive"]) for label in matched_labels
        )
        sql_mismatches = sum(
            compact_sql(row.get("candidate sql") or row.get("candidate_sql"))
            != compact_sql(label.get("sql"))
            for row, label in zip(rows, matched_labels)
        )
        if sql_mismatches:
            raise ValueError(
                f"{pool}: {sql_mismatches} candidate SQL rows do not match "
                "the frozen final reference"
            )
        input_manifest[pool] = {
            "dataset": dataset,
            "path": str(path),
            "rows": len(rows),
            "positives": positive_count,
            "negatives": len(rows) - positive_count,
            "final_label_path": str(FINAL_LABELS),
        }
        work.extend((pool, dataset, row) for row in rows)
    if args.limit:
        work = work[: args.limit]

    registry_cache: dict[tuple[str, str], SchemaRegistry] = {}
    profiled_registry_cache: dict[tuple[str, str], SchemaRegistry] = {}
    traces: list[NaturalTrace] = []
    for index, (pool, dataset, row) in enumerate(work, start=1):
        db_id = str(row.get("db_id") or "")
        cache_key = (dataset, db_id)
        if cache_key not in registry_cache:
            registry_path = DEFAULT_REGISTRIES[dataset] / f"{db_id}.yaml"
            registry_cache[cache_key] = SchemaRegistry.from_yaml(registry_path)
        registry = registry_cache[cache_key]
        sql = row.get("candidate sql") or row.get("candidate_sql") or ""
        if dataset == "bird" and sql:
            profile_key = (db_id, compact_sql(sql))
            if profile_key not in profiled_registry_cache:
                profiled_registry_cache[profile_key] = (
                    registry_with_filter_uniqueness_profile(
                        registry,
                        BIRD_DATABASES / db_id / f"{db_id}.sqlite",
                        sql,
                        decision_eligible=False,
                    )
                )
            registry = profiled_registry_cache[profile_key]
        traces.append(
            NaturalTrace(
                pool=pool,
                dataset=dataset,
                label=bool(
                    final_labels[
                        label_key(dataset, pool, row.get("question_id"))
                    ]["final_paper_core_positive"]
                ),
                trace=trace_row(
                    dataset,
                    row,
                    registry,
                    guard_static_pass_violations=False,
                ),
            )
        )
        if index % 50 == 0 or index == len(work):
            print(f"traced {index}/{len(work)} rows", flush=True)

    cohorts: dict[str, list[NaturalTrace]] = {}
    for item in traces:
        for name in cohort_names(item):
            cohorts.setdefault(name, []).append(item)

    exhaustive_detectable: dict[str, set[str]] = {}
    for cohort, rows in cohorts.items():
        exhaustive_detectable[cohort] = {
            f"{item.pool}:{item.trace.sample_id}"
            for item in rows
            if item.label
            and classify_at_budget(
                item.trace,
                None,
                evidence_scope_policy=args.evidence_scope_policy,
            ).detected
        }

    cohort_order = (
        "overall",
        "bird",
        "spider",
        "bird_glm",
        "bird_qwen",
        "spider_glm",
        "spider_qwen",
    )
    summaries = {
        cohort: [
            summarize_cohort(
                cohorts[cohort],
                budget,
                exhaustive_detectable[cohort],
                evidence_scope_policy=args.evidence_scope_policy,
            )
            for budget in budgets
        ]
        for cohort in cohort_order
        if cohort in cohorts
    }
    report = {
        "experiment": "natural_candidate_evidence_budget_sensitivity_v1",
        "definition": {
            "reporting_scope": (
                "primary witness-closed candidate evaluation with budget-sensitive "
                "dynamic evidence and obligation-local static discharge; every "
                "reported dynamic VIOLATION requires an executable "
                "contract-preserving witness"
            ),
            "budget_unit": "numeric value-profile executions per dynamic plan",
            "ordered_value_profiles": [
                {"multiplier": multiplier, "offset": offset}
                for multiplier, offset in VALUE_PROFILES
            ],
            "all_definition": (
                "all ten ordered profiles, or fewer after exact deduplication"
            ),
            "positive_definition": (
                "independently frozen candidate_grain_risk_label.risk_violation"
            ),
            "positive_retention_definition": (
                "label-positive rows detected within the budget prefix divided "
                "by label-positive rows detected by all profiles"
            ),
            "label_blinding": (
                "labels are not passed to contract validation, signature "
                "extraction, plan matching, pair construction, or execution"
            ),
            "evidence_scope": (
                "Witness admissibility is assigned before pair construction. "
                "TST-only and candidate-instance-only uniqueness alarms are "
                "deferred under the supported policy."
            ),
            "evidence_scope_policy": args.evidence_scope_policy,
            "static_guard": (
                "No global static PASS may override an executable dynamic VIOLATION; "
                "static discharge is obligation-local"
            ),
            "profile_limitation": (
                "profiles are affine numeric transformations of one structural "
                "pair; they do not vary join topology, row multiplicity pattern, "
                "null placement, grouping cardinality, or tie structure"
            ),
            "early_stop": "stop each dynamic plan at its first VIOLATION",
        },
        "inputs": input_manifest,
        "evaluated_rows": len(traces),
        "cohorts": summaries,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "natural_budget_sensitivity.summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    table_path = args.output_dir / "natural_budget_sensitivity.table.tsv"
    write_table(table_path, summaries)
    trace_path = args.output_dir / "natural_budget_sensitivity.traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for item in traces:
            decision = classify_at_budget(
                item.trace,
                None,
                evidence_scope_policy=args.evidence_scope_policy,
            )
            handle.write(
                json.dumps(
                    {
                        "pool": item.pool,
                        "dataset": item.dataset,
                        "label": item.label,
                        "sample_id": item.trace.sample_id,
                        "static_violation": item.trace.static_violation,
                        "static_pass": item.trace.static_pass,
                        "static_guard_status": item.trace.static_guard_status,
                        "unsupported_reason": item.trace.unsupported_reason,
                        "decision_status": decision.status,
                        "evidence_scope_policy": args.evidence_scope_policy,
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
                            for plan in item.trace.plan_traces
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"summary: {summary_path}", file=sys.stderr)
    print(f"table: {table_path}", file=sys.stderr)
    print(f"traces: {trace_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
