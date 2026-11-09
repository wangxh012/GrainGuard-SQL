#!/usr/bin/env python3
"""Summarize predeclared evidence-scope decisions and their ablation views.

The filename is retained for compatibility with existing experiment commands.
Unlike the former two-replay analysis, this program consumes one trace whose
plans were assigned witness admissibility before construction and execution.
It computes metrics only; it does not alter any query-level decision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.evidence_scope import (  # noqa: E402
    DEFER_TST_POLICY,
    SUPPORTED_POLICY,
    UNRESTRICTED_POLICY,
    recorded_trace_status,
)


POLICIES = (
    UNRESTRICTED_POLICY,
    DEFER_TST_POLICY,
    SUPPORTED_POLICY,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decision_class(row: dict[str, Any]) -> str:
    supported = recorded_trace_status(row, SUPPORTED_POLICY)
    if supported == "VIOLATION":
        return "CONFIRMED_VIOLATION"
    unrestricted = recorded_trace_status(row, UNRESTRICTED_POLICY)
    if unrestricted != "VIOLATION":
        return supported
    after_tst = recorded_trace_status(row, DEFER_TST_POLICY)
    if after_tst == "VIOLATION":
        return "DEFERRED_INSTANCE_ONLY_UNIQUENESS"
    return "DEFERRED_TST_ONLY"


def metrics(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    statuses = [recorded_trace_status(row, policy) for row in rows]
    labels = [bool(row["label"]) for row in rows]
    predictions = [status == "VIOLATION" for status in statuses]
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    tn = len(rows) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    decided = sum(status in {"PASS", "VIOLATION"} for status in statuses)
    return {
        "policy": policy,
        "n": len(rows),
        "positive": sum(labels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "decided": decided,
        "decision_coverage": decided / len(rows) if rows else None,
        "status_counts": dict(sorted(Counter(statuses).items())),
    }


def pct(value: float | None) -> str:
    return "" if value is None else f"{100 * value:.2f}"


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.traces)
    if len(rows) != 1160:
        raise ValueError("trace must contain the closed 1,160-candidate V2.3 cohort")
    if any(
        "evidence_scope" not in plan
        for row in rows
        for plan in row.get("plans") or []
    ):
        raise ValueError("trace predates pre-execution evidence-scope assignment")

    ledger = [
        {
            "pool": row["pool"],
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "label": bool(row["label"]),
            "operator": str(row["sample_id"]).rsplit(":", 2)[-2],
            "decision_status": recorded_trace_status(row, SUPPORTED_POLICY),
            "decision_class": decision_class(row),
        }
        for row in rows
    ]
    rows_by_scope: dict[str, list[dict[str, Any]]] = {"overall": rows}
    for pool in ("bird_glm", "bird_qwen", "spider_glm", "spider_qwen"):
        rows_by_scope[pool] = [row for row in rows if row["pool"] == pool]
    scored = [
        {"scope": scope, **metrics(scope_rows, policy)}
        for scope, scope_rows in rows_by_scope.items()
        for policy in POLICIES
    ]

    report = {
        "experiment": "v23_predeclared_evidence_scope",
        "policy": {
            "assignment_time": "before witness construction and execution",
            "supported": "Only supported-scope witnesses confirm VIOLATION.",
            "deferred_tst_only": "TST-only alarms return ABSTAIN.",
            "deferred_instance_only_uniqueness": (
                "Alarms whose plan is discharged only by observed, undeclared "
                "uniqueness return ABSTAIN."
            ),
        },
        "input": {"path": args.traces.as_posix(), "sha256": sha256(args.traces)},
        "decision_class_counts": dict(
            sorted(Counter(row["decision_class"] for row in ledger).items())
        ),
        "rows": scored,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evidence_scope.summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "evidence_scope.ledger.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in ledger:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    fields = (
        "scope", "policy", "n", "positive", "tp", "fp", "fn", "tn",
        "precision_pct", "recall_pct", "f1_pct", "decided",
        "decision_coverage_pct",
    )
    with (args.output_dir / "evidence_scope.table.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in scored:
            writer.writerow(
                {
                    **{key: row[key] for key in fields[:8]},
                    "precision_pct": pct(row["precision"]),
                    "recall_pct": pct(row["recall"]),
                    "f1_pct": pct(row["f1"]),
                    "decided": row["decided"],
                    "decision_coverage_pct": pct(row["decision_coverage"]),
                }
            )

    print(f"evidence-scope rows: {len(ledger)}")


if __name__ == "__main__":
    main()
