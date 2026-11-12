#!/usr/bin/env python3
"""Replay frozen natural candidates with Contract V2.1 and the paper search policy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORTING = ROOT / "workflows/05_reporting"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REPORTING) not in sys.path:
    sys.path.insert(0, str(REPORTING))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = load_module("v23_natural_budget", REPORTING / "run_natural_budget_sensitivity.py")

from grain_test_sql.evidence_scope import (  # noqa: E402
    EVIDENCE_SCOPE_POLICIES,
    SUPPORTED_POLICY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--primary-labels",
        default="artifacts/annotations/sql_labels/final_reference_labels.paper_core.v2.jsonl",
    )
    parser.add_argument(
        "--supplemental-labels",
        default="artifacts/annotations/sql_labels/supplemental_spec_failures.final.jsonl",
    )
    parser.add_argument("--budgets", default="1,3,6,10,all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--evidence-scope-policy",
        choices=EVIDENCE_SCOPE_POLICIES,
        default=SUPPORTED_POLICY,
        help=(
            "Witness-admissibility policy fixed before execution; supported is "
            "the paper-facing decision rule."
        ),
    )
    parser.add_argument(
        "--output-comparator",
        choices=("exact", "order_normalized", "numeric_tolerant"),
        default="exact",
        help="Projected-output comparison policy; exact remains the primary setting.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def display_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), ROOT)).as_posix()


def compact_sql(sql: Any) -> str:
    return re.sub(r"\s+", "", str(sql or ""))


def canonical_model(value: Any) -> str:
    aliases = {
        "glm": "glm-5.1",
        "glm-5.1": "glm-5.1",
        "qwen": "qwen3.6-27b",
        "qwen3.6-27b": "qwen3.6-27b",
    }
    key = str(value or "").casefold()
    if key not in aliases:
        raise ValueError(f"unsupported model: {value!r}")
    return aliases[key]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_labels(paths: list[Path]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    labels: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for path in paths:
        for row in V1.read_jsonl(path):
            if row.get("sql_role") != "candidate_sql":
                continue
            key = (
                str(row["dataset"]).casefold(),
                canonical_model(row["model"]),
                int(row["question_id"]),
                str(row["db_id"]),
            )
            if key in labels:
                raise ValueError(f"duplicate frozen label: {key}")
            labels[key] = row
    return labels


def main() -> None:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    output_dir = resolve(args.output_dir)
    label_paths = [resolve(args.primary_labels), resolve(args.supplemental_labels)]
    labels = read_labels(label_paths)
    budgets = V1.parse_budgets(args.budgets)
    pools = (
        ("bird_glm", "bird", "glm-5.1"),
        ("bird_qwen", "bird", "qwen3.6-27b"),
        ("spider_glm", "spider", "glm-5.1"),
        ("spider_qwen", "spider", "qwen3.6-27b"),
    )
    work: list[tuple[str, str, str, dict[str, Any]]] = []
    input_manifest: dict[str, Any] = {}
    for pool, dataset, model in pools:
        path = input_dir / f"{pool}_candidate.v23.jsonl"
        rows = V1.read_jsonl(path)
        expected = 322 if dataset == "bird" else 258
        if len(rows) != expected:
            raise ValueError(f"{pool}: expected {expected} rows, found {len(rows)}")
        for row in rows:
            key = (dataset, model, int(row["question_id"]), str(row["db_id"]))
            label = labels.get(key)
            if label is None:
                raise ValueError(f"missing frozen label: {key}")
            if compact_sql(row.get("candidate sql")) != compact_sql(
                label.get("candidate_sql") or label.get("sql")
            ):
                raise ValueError(f"candidate SQL differs from frozen reviewed SQL: {key}")
            work.append((pool, dataset, model, row))
        input_manifest[pool] = {
            "path": display_path(path),
            "rows": len(rows),
            "sha256": sha256(path),
        }
    if len(work) != 1160:
        raise ValueError(f"expected 1,160 V2.3 candidate rows, found {len(work)}")
    if args.limit:
        work = work[: args.limit]

    registry_cache: dict[tuple[str, str], Any] = {}
    profiled_registry_cache: dict[tuple[str, str], Any] = {}
    traces: list[Any] = []
    for index, (pool, dataset, model, row) in enumerate(work, start=1):
        db_id = str(row["db_id"])
        cache_key = (dataset, db_id)
        if cache_key not in registry_cache:
            registry_path = V1.DEFAULT_REGISTRIES[dataset] / f"{db_id}.yaml"
            registry_cache[cache_key] = V1.SchemaRegistry.from_yaml(registry_path)
        registry = registry_cache[cache_key]
        sql = row.get("candidate sql") or ""
        if dataset == "bird" and sql:
            profile_key = (db_id, compact_sql(sql))
            if profile_key not in profiled_registry_cache:
                profiled_registry_cache[profile_key] = V1.registry_with_filter_uniqueness_profile(
                    registry,
                    V1.BIRD_DATABASES / db_id / f"{db_id}.sqlite",
                    sql,
                    decision_eligible=False,
                )
            registry = profiled_registry_cache[profile_key]
        key = (dataset, model, int(row["question_id"]), db_id)
        label = labels[key]
        traces.append(
            V1.NaturalTrace(
                pool=pool,
                dataset=dataset,
                label=bool(label["final_paper_core_positive"]),
                trace=V1.trace_row(
                    dataset,
                    row,
                    registry,
                    guard_static_pass_violations=False,
                    output_comparator=args.output_comparator,
                ),
            )
        )
        if index % 50 == 0 or index == len(work):
            print(f"traced {index}/{len(work)} rows", flush=True)

    cohorts: dict[str, list[Any]] = {}
    for item in traces:
        for name in V1.cohort_names(item):
            cohorts.setdefault(name, []).append(item)
    exhaustive_detectable = {
        cohort: {
            f"{item.pool}:{item.trace.sample_id}"
            for item in rows
            if item.label
            and V1.classify_at_budget(
                item.trace,
                None,
                evidence_scope_policy=args.evidence_scope_policy,
            ).detected
        }
        for cohort, rows in cohorts.items()
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
            V1.summarize_cohort(
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
        "experiment": "natural_candidate_validation_v2.3_contract_v2.1",
        "contract_policy": "Frozen Contract V2.1; no schema, prompt, gate, or label changes.",
        "selection_policy": "All 580 accepted V2.1 questions crossed with both frozen candidate models.",
        "search_policy": "Ten ordered value profiles with witness closure; witness admissibility is assigned before pair construction and execution.",
        "evidence_scope_policy": args.evidence_scope_policy,
        "output_comparator": args.output_comparator,
        "labels_used_after_replay": True,
        "instance_uniqueness_role": "pre-execution witness-scope assignment only",
        "inputs": input_manifest,
        "label_sources": [
            {"path": display_path(path), "sha256": sha256(path)}
            for path in label_paths
        ],
        "evaluated_rows": len(traces),
        "cohorts": summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "natural_candidate_v23.summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    V1.write_table(output_dir / "natural_candidate_v23.table.tsv", summaries)
    trace_path = output_dir / "natural_candidate_v23.traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for item in traces:
            decision = V1.classify_at_budget(
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
                        "decision_reason": (
                            "CONFIRMED_WITNESS"
                            if decision.status == "VIOLATION"
                            else "WITNESS_OUTSIDE_SUPPORTED_SCOPE"
                            if decision.status == "ABSTAIN"
                            else None
                        ),
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
    print(f"summary: {summary_path}")
    print(f"table: {output_dir / 'natural_candidate_v23.table.tsv'}")
    print(f"traces: {trace_path}")


if __name__ == "__main__":
    main()
