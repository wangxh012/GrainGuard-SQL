#!/usr/bin/env python3
"""Score frozen BIRD baselines on their strict Contract-V2.1 intersection.

The archived baseline outputs omit 15 of the 322 accepted BIRD questions for
each generator.  This script therefore does not impute or silently enlarge the
baseline cohort: it intersects the question identifiers covered by every
reported method and labels the resulting comparison as secondary.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.evidence_scope import emitted_trace_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--llm-glm-judge-bird-glm", default="")
    parser.add_argument("--llm-glm-judge-bird-qwen", default="")
    parser.add_argument("--llm-qwen-judge-bird-glm", default="")
    parser.add_argument("--llm-qwen-judge-bird-qwen", default="")
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(path: Path) -> dict[int, dict[str, Any]]:
    return {
        int(row["question_id"]): row
        for row in read_jsonl(path)
        if row.get("question_id") is not None
    }


def compact_sql(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def configured_llm_paths(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    configured: dict[str, dict[str, Path]] = {}
    for method, values in (
        (
            "llm_self_check_glm_judge",
            {
                "bird_glm": args.llm_glm_judge_bird_glm,
                "bird_qwen": args.llm_glm_judge_bird_qwen,
            },
        ),
        (
            "llm_self_check_qwen_judge",
            {
                "bird_glm": args.llm_qwen_judge_bird_glm,
                "bird_qwen": args.llm_qwen_judge_bird_qwen,
            },
        ),
    ):
        supplied = {pool: value for pool, value in values.items() if value}
        if supplied and len(supplied) != len(values):
            missing = sorted(set(values) - set(supplied))
            raise ValueError(
                f"{method}: both generator-pool artifacts are required; missing {missing}"
            )
        if supplied:
            configured[method] = {
                pool: resolve(value) for pool, value in supplied.items()
            }
    return configured


def validate_llm_rows(
    *,
    method: str,
    pool: str,
    rows: dict[int, dict[str, Any]],
    candidate_rows: dict[int, dict[str, Any]],
    required_ids: list[int],
) -> None:
    missing = sorted(set(required_ids) - set(rows))
    if missing:
        raise ValueError(f"{method}/{pool}: missing {len(missing)} shared-intersection rows")
    for question_id in required_ids:
        row = rows[question_id]
        policy = row.get("input_policy") or {}
        schema_scope = row.get("schema_scope") or policy.get("schema_scope")
        if (
            schema_scope != "contract-linked"
            or policy.get("schema_source") != "contract_schema_linking"
            or policy.get("reference_sql_derived_table_scope") is not False
        ):
            raise ValueError(
                f"{method}/{pool}/qid={question_id}: LLM result is not provenance-clean "
                "contract-linked output"
            )
        if compact_sql(row.get("candidate_sql")) != compact_sql(
            candidate_rows[question_id].get("candidate sql")
            or candidate_rows[question_id].get("candidate_sql")
        ):
            raise ValueError(
                f"{method}/{pool}/qid={question_id}: candidate SQL differs from V2.3 input"
            )


def validate_legacy_llm_rows(
    *,
    method: str,
    pool: str,
    rows: dict[int, dict[str, Any]],
    candidate_rows: dict[int, dict[str, Any]],
    required_ids: list[int],
) -> None:
    missing = sorted(set(required_ids) - set(rows))
    if missing:
        raise ValueError(f"{method}/{pool}: missing {len(missing)} shared-intersection rows")
    for question_id in required_ids:
        if compact_sql(rows[question_id].get("candidate_sql")) != compact_sql(
            candidate_rows[question_id].get("candidate sql")
            or candidate_rows[question_id].get("candidate_sql")
        ):
            raise ValueError(
                f"{method}/{pool}/qid={question_id}: candidate SQL differs from V2.3 input"
            )


def trace_status(row: dict[str, Any]) -> str:
    return emitted_trace_status(row)


def score(labels: list[bool], predictions: list[bool], supported: list[bool]) -> dict[str, Any]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {
        "n": len(labels),
        "positive": sum(labels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": sum(supported) / len(supported),
    }


def pct(value: float | None) -> str:
    return "" if value is None else f"{100 * value:.2f}"


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    llm_paths = configured_llm_paths(args)

    traces_by_pool: dict[str, dict[int, dict[str, Any]]] = {}
    for row in read_jsonl(run_dir / "replay/natural_candidate_v23.traces.jsonl"):
        pool = str(row["pool"])
        if not pool.startswith("bird_"):
            continue
        question_id = int(str(row["sample_id"]).rsplit(":", 4)[2])
        traces_by_pool.setdefault(pool, {})[question_id] = row

    method_labels = {
        "grain_guard_v23": "GrainGuard-SQL",
        "original_execution_matching": "Original-DB execution match",
        "static_grain_screening": "Static grain screening",
        "test_suite_adaptation": "TSA adaptation",
        "sqlhd_inspired": "SQLHD-inspired MR",
        "sqldriller_inspired": "SQLDriller-inspired separation",
        "spotit_plus_official": "SpotIt+ official",
        "legacy_llm_self_check_glm": "LLM self-check (GLM; legacy scope)",
        "legacy_llm_self_check_qwen": "LLM self-check (Qwen; legacy scope)",
    }
    if "llm_self_check_glm_judge" in llm_paths:
        method_labels["llm_self_check_glm_judge"] = "LLM self-check (GLM-5.1 judge)"
    if "llm_self_check_qwen_judge" in llm_paths:
        method_labels["llm_self_check_qwen_judge"] = "LLM self-check (Qwen judge)"
    output_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    eval_scope = keyed(ROOT / "artifacts/datasets/bird/eval/bird_eval_set.jsonl")
    linking_scope = keyed(
        ROOT / "artifacts/runs/bird/shared/00_generation_inputs/schema_linking.jsonl"
    )
    legacy_scope_counts = {"same": 0, "legacy_strict_superset": 0, "other": 0}

    for pool, run_model, filename_model in (
        ("bird_glm", "glm-5.1", "glm"),
        ("bird_qwen", "qwen", "qwen"),
    ):
        traces = traces_by_pool[pool]
        sources = {
            "original_execution_matching": keyed(
                ROOT / f"artifacts/runs/original_db_execution_match/bird/{filename_model}_candidate.original_db_exec_match.jsonl"
            ),
            "static_grain_screening": keyed(
                ROOT / f"artifacts/runs/static_grain_screening/bird/{filename_model}_candidate.static_grain_screening.jsonl"
            ),
            "test_suite_adaptation": keyed(
                ROOT / f"artifacts/runs/bird/{run_model}/05_baselines/test_suite_style_reasoning_full.extra_scope.jsonl"
            ),
            "sqlhd_inspired": keyed(
                ROOT / f"artifacts/runs/bird/{run_model}/05_baselines/sqlhd_style_reasoning_sql.jsonl"
            ),
            "sqldriller_inspired": keyed(
                ROOT / f"artifacts/runs/bird/{run_model}/05_baselines/sql_driller_style_reasoning_sql_targeted_attempts300.jsonl"
            ),
            "spotit_plus_official": keyed(
                ROOT / f"artifacts/runs/bird/{run_model}/05_baselines/spotit_plus_official_reference.jsonl"
            ),
            "legacy_llm_self_check_glm": keyed(
                ROOT
                / (
                    "artifacts/runs/bird/glm-5.1/05_baselines/llm_self_check_reasoning_sql.jsonl"
                    if pool == "bird_glm"
                    else "artifacts/runs/bird/qwen/05_baselines/llm_self_check_reasoning_sql.glm_judge.current_contracts.jsonl"
                )
            ),
            "legacy_llm_self_check_qwen": keyed(
                ROOT
                / (
                    "artifacts/runs/bird/glm-5.1/05_baselines/llm_self_check_reasoning_sql.qwen_judge.jsonl"
                    if pool == "bird_glm"
                    else "artifacts/runs/bird/qwen/05_baselines/llm_self_check_reasoning_sql.qwen_judge.jsonl"
                )
            ),
        }
        common = set(traces)
        for rows in sources.values():
            common &= set(rows)
        question_ids = sorted(common)
        omitted = sorted(set(traces) - common)
        if len(question_ids) != 307 or len(omitted) != 15:
            raise ValueError(
                f"{pool}: expected strict intersection 307 and 15 omissions; "
                f"found {len(question_ids)} and {len(omitted)}"
            )
        cohort_rows.append(
            {"pool": pool, "target_questions": len(traces), "intersection_questions": len(question_ids), "omitted_question_ids": omitted}
        )
        labels = [bool(traces[qid]["label"]) for qid in question_ids]
        candidate_rows = keyed(run_dir / "inputs" / f"{pool}_candidate.v23.jsonl")
        for question_id in question_ids:
            legacy_tables = set(
                eval_scope[question_id].get("scoped_tables")
                or eval_scope[question_id].get("sql_tables")
                or []
            )
            linked_tables = set(
                linking_scope[question_id].get("schema_linking_tables") or []
            )
            if legacy_tables == linked_tables:
                legacy_scope_counts["same"] += 1
            elif linked_tables < legacy_tables:
                legacy_scope_counts["legacy_strict_superset"] += 1
            else:
                legacy_scope_counts["other"] += 1

        method_values: dict[str, tuple[list[bool], list[bool]]] = {}
        v23_status = [trace_status(traces[qid]) for qid in question_ids]
        method_values["grain_guard_v23"] = (
            [status == "VIOLATION" for status in v23_status],
            [status in {"PASS", "VIOLATION"} for status in v23_status],
        )

        original = sources["original_execution_matching"]
        method_values["original_execution_matching"] = (
            [bool(original[qid]["prediction"]["predicted_grain_error"]) for qid in question_ids],
            [(original[qid].get("test_suite_result") or {}).get("status") != "UNSUPPORTED" for qid in question_ids],
        )
        static = sources["static_grain_screening"]
        method_values["static_grain_screening"] = (
            [bool((static[qid].get("static_grain_result") or {}).get("is_alert")) for qid in question_ids],
            [(static[qid].get("static_grain_result") or {}).get("status") in {"PASS", "ALERT", "VIOLATION"} for qid in question_ids],
        )
        tsa = sources["test_suite_adaptation"]
        tsa_results = [(tsa[qid].get("test_suite_result") or {}) for qid in question_ids]
        method_values["test_suite_adaptation"] = (
            [row.get("status") == "FAIL" and not row.get("candidate_errors") for row in tsa_results],
            [row.get("status") != "UNSUPPORTED" for row in tsa_results],
        )
        sqlhd = sources["sqlhd_inspired"]
        sqlhd_results = [(sqlhd[qid].get("sqlhd_style_result") or {}) for qid in question_ids]
        method_values["sqlhd_inspired"] = (
            [
                row.get("status") == "VIOLATION"
                and not row.get("error")
                and not any(
                    check.get("status") == "VIOLATION"
                    and "parse error" in str(check.get("diagnosis") or "").lower()
                    for check in row.get("checks") or []
                )
                for row in sqlhd_results
            ],
            [row.get("status") != "UNSUPPORTED" for row in sqlhd_results],
        )
        driller = sources["sqldriller_inspired"]
        driller_results = [(driller[qid].get("sql_driller_style_result") or {}) for qid in question_ids]
        method_values["sqldriller_inspired"] = (
            [row.get("status") == "VIOLATION" for row in driller_results],
            [row.get("status") in {"PASS", "VIOLATION"} for row in driller_results],
        )
        spotit = sources["spotit_plus_official"]
        method_values["spotit_plus_official"] = (
            [str(spotit[qid].get("status")) == "VIOLATION" for qid in question_ids],
            [str(spotit[qid].get("status")) in {"PASS", "VIOLATION"} for qid in question_ids],
        )
        for method in ("legacy_llm_self_check_glm", "legacy_llm_self_check_qwen"):
            legacy_rows = sources[method]
            validate_legacy_llm_rows(
                method=method,
                pool=pool,
                rows=legacy_rows,
                candidate_rows=candidate_rows,
                required_ids=question_ids,
            )
            decisions = [
                legacy_rows[qid].get("llm_self_check") or {}
                for qid in question_ids
            ]
            method_values[method] = (
                [decision.get("has_grain_risk") is True for decision in decisions],
                [
                    not legacy_rows[qid].get("parse_error")
                    and isinstance(decision.get("has_grain_risk"), bool)
                    for qid, decision in zip(question_ids, decisions)
                ],
            )
        for method, paths_by_pool in llm_paths.items():
            llm_rows = keyed(paths_by_pool[pool])
            validate_llm_rows(
                method=method,
                pool=pool,
                rows=llm_rows,
                candidate_rows=candidate_rows,
                required_ids=question_ids,
            )
            decisions = [llm_rows[qid].get("llm_self_check") or {} for qid in question_ids]
            method_values[method] = (
                [decision.get("has_grain_risk") is True for decision in decisions],
                [
                    not llm_rows[qid].get("parse_error")
                    and isinstance(decision.get("has_grain_risk"), bool)
                    for qid, decision in zip(question_ids, decisions)
                ],
            )

        for method, (predictions, supported) in method_values.items():
            result = score(labels, predictions, supported)
            output_rows.append(
                {"pool": pool, "method": method, "method_label": method_labels[method], **result}
            )

    pooled_rows: list[dict[str, Any]] = []
    for method in method_labels:
        rows = [row for row in output_rows if row["method"] == method]
        tp = sum(int(row["tp"]) for row in rows)
        fp = sum(int(row["fp"]) for row in rows)
        fn = sum(int(row["fn"]) for row in rows)
        tn = sum(int(row["tn"]) for row in rows)
        n = tp + fp + fn + tn
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
        pooled_rows.append(
            {
                "pool": "bird_pooled",
                "method": method,
                "method_label": method_labels[method],
                "n": n,
                "positive": sum(int(row["positive"]) for row in rows),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "coverage": sum(float(row["coverage"]) * int(row["n"]) for row in rows) / n,
            }
        )
    output_rows.extend(pooled_rows)

    report = {
        "protocol": "v23_secondary_supported_scope_shared_intersection_bird_baselines",
        "interpretation": "Descriptive secondary comparison on the 307 accepted BIRD questions per generator covered by every method; it is not the primary 1,160-row V2.3 estimand.",
        "cohorts": cohort_rows,
        "legacy_llm_scope_audit": {
            "interpretation": (
                "Archived self-check prompts used eval-set scoped_tables rather than "
                "the reference-free contract schema-linking artifact."
            ),
            **legacy_scope_counts,
        },
        "llm_self_check_artifacts": {
            method: {pool: str(path) for pool, path in paths.items()}
            for method, paths in llm_paths.items()
        },
        "rows": output_rows,
    }
    (out_dir / "v23_shared_intersection_baselines.summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = ["pool", "method", "method_label", "n", "positive", "tp", "fp", "fn", "tn", "precision_pct", "recall_pct", "f1_pct", "coverage_pct"]
    with (out_dir / "v23_shared_intersection_baselines.table.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fields[:9]},
                    "precision_pct": pct(row["precision"]),
                    "recall_pct": pct(row["recall"]),
                    "f1_pct": pct(row["f1"]),
                    "coverage_pct": pct(row["coverage"]),
                }
            )


if __name__ == "__main__":
    main()
