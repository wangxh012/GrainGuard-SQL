#!/usr/bin/env python3
"""Audit whether frozen baseline outputs cover the Contract V2.1 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASELINES = {
    "original_execution_matching": {
        "bird_glm": "artifacts/runs/original_db_execution_match/bird/glm_candidate.original_db_exec_match.jsonl",
        "bird_qwen": "artifacts/runs/original_db_execution_match/bird/qwen_candidate.original_db_exec_match.jsonl",
        "spider_glm": "artifacts/runs/original_db_execution_match/spider/glm_candidate.original_db_exec_match.jsonl",
        "spider_qwen": "artifacts/runs/original_db_execution_match/spider/qwen_candidate.original_db_exec_match.jsonl",
    },
    "static_grain_screening": {
        "bird_glm": "artifacts/runs/static_grain_screening/bird/glm_candidate.static_grain_screening.jsonl",
        "bird_qwen": "artifacts/runs/static_grain_screening/bird/qwen_candidate.static_grain_screening.jsonl",
        "spider_glm": "artifacts/runs/static_grain_screening/spider/glm_candidate.static_grain_screening.jsonl",
        "spider_qwen": "artifacts/runs/static_grain_screening/spider/qwen_candidate.static_grain_screening.jsonl",
    },
    "test_suite_adaptation": {
        "bird_glm": "artifacts/runs/bird/glm-5.1/05_baselines/test_suite_style_reasoning_full.extra_scope.jsonl",
        "bird_qwen": "artifacts/runs/bird/qwen/05_baselines/test_suite_style_reasoning_full.extra_scope.jsonl",
        "spider_glm": "artifacts/runs/spider/glm-5.1/05_baselines/test_suite_style_reasoning_small_only.extra_scope.jsonl",
        "spider_qwen": "artifacts/runs/spider/qwen/05_baselines/test_suite_style_reasoning_small_only.extra_scope.jsonl",
    },
    "sqlhd_inspired": {
        "bird_glm": "artifacts/runs/bird/glm-5.1/05_baselines/sqlhd_style_reasoning_sql.jsonl",
        "bird_qwen": "artifacts/runs/bird/qwen/05_baselines/sqlhd_style_reasoning_sql.jsonl",
        "spider_glm": "artifacts/runs/spider/glm-5.1/05_baselines/sqlhd_style_reasoning_sql.jsonl",
        "spider_qwen": "artifacts/runs/spider/qwen/05_baselines/sqlhd_style_reasoning_sql.jsonl",
    },
    "sqldriller_inspired": {
        "bird_glm": "artifacts/runs/bird/glm-5.1/05_baselines/sql_driller_style_reasoning_sql_targeted_attempts300.jsonl",
        "bird_qwen": "artifacts/runs/bird/qwen/05_baselines/sql_driller_style_reasoning_sql_targeted_attempts300.jsonl",
        "spider_glm": "artifacts/runs/spider/glm-5.1/05_baselines/sql_driller_style_reasoning_sql_targeted_attempts300.jsonl",
        "spider_qwen": "artifacts/runs/spider/qwen/05_baselines/sql_driller_style_reasoning_sql_targeted_attempts300.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    input_dir = resolve(args.input_dir)
    target_qids = {
        pool: {int(row["question_id"]) for row in read_jsonl(input_dir / f"{pool}_candidate.v23.jsonl")}
        for pool in ("bird_glm", "bird_qwen", "spider_glm", "spider_qwen")
    }
    rows: list[dict[str, Any]] = []
    for baseline, pools in BASELINES.items():
        for pool, raw_path in pools.items():
            path = resolve(raw_path)
            baseline_qids = (
                {int(row["question_id"]) for row in read_jsonl(path) if row.get("question_id") is not None}
                if path.exists()
                else set()
            )
            target = target_qids[pool]
            rows.append(
                {
                    "baseline": baseline,
                    "pool": pool,
                    "path": path.relative_to(ROOT).as_posix(),
                    "artifact_exists": path.exists(),
                    "target_rows": len(target),
                    "covered_rows": len(target & baseline_qids),
                    "missing_rows": len(target - baseline_qids),
                    "missing_question_ids": sorted(target - baseline_qids),
                    "full_v23_comparison_valid": target <= baseline_qids,
                }
            )
    report = {
        "protocol": "v2.3_frozen_baseline_cohort_coverage_audit",
        "interpretation": (
            "A baseline may be compared on the full Contract V2.1 cohort only if every "
            "target question is covered. Otherwise rerun the missing rows or declare a "
            "shared-intersection secondary comparison."
        ),
        "all_full_v23_comparisons_valid": all(row["full_v23_comparison_valid"] for row in rows),
        "rows": rows,
    }
    output = resolve(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
