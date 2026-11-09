"""Summarize an unfiltered Contract V2.1 executor/validation V2.2 replay."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OPERATORS = ("GRT", "CGR", "DET", "EST", "ART", "TST")
VERDICT_STATUSES = {"PASS", "VIOLATION"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def replay_key(row: dict[str, Any]) -> str:
    if row.get("replay_key"):
        return str(row["replay_key"])
    return "|".join(
        map(
            str,
            (row["question_id"], row["db_id"], row["contract_type"], int(row.get("contract_index", 0))),
        )
    )


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def count_row(rows: list[dict[str, Any]], dataset: str | None, operator: str | None) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if (dataset is None or row["dataset"] == dataset)
        and (operator is None or row["contract_type"] == operator)
    ]
    n = len(selected)
    fault_compatible = sum(row["fault_status"] in VERDICT_STATUSES for row in selected)
    control_compatible = sum(row["certified_control_status"] in VERDICT_STATUSES for row in selected)
    pair_compatible = sum(
        row["fault_status"] in VERDICT_STATUSES
        and row["certified_control_status"] in VERDICT_STATUSES
        for row in selected
    )
    triggers = sum(row["fault_status"] == "VIOLATION" for row in selected)
    raw_control_triggers = sum(row["raw_control_status"] == "VIOLATION" for row in selected)
    certified_negative = len(selected)
    certified_false_triggers = sum(
        row["certified_control_status"] == "VIOLATION" for row in selected
    )
    contract_positive = sum(row["raw_control_certification"] == "CONTRACT_POSITIVE" for row in selected)
    ambiguous_controls = sum(row["raw_control_certification"] == "AMBIGUOUS" for row in selected)
    witness_records = sum(row["triggered"] and row["reconstruction_status"] == "RECONSTRUCTED" for row in selected)
    witnesses_valid = sum(bool(row.get("code_audit_valid")) for row in selected if row["triggered"])
    return {
        "dataset": dataset or "All",
        "operator": operator or "Total",
        "paired_variants": n,
        "fault_executor_compatible": fault_compatible,
        "fault_executor_compatibility": rate(fault_compatible, n),
        "control_executor_compatible": control_compatible,
        "control_executor_compatibility": rate(control_compatible, n),
        "pair_executor_compatible": pair_compatible,
        "pair_executor_compatibility": rate(pair_compatible, n),
        "fault_triggers": triggers,
        "trigger_rate_full_pool": rate(triggers, n),
        "trigger_rate_among_fault_compatible": rate(triggers, fault_compatible),
        "raw_control_triggers": raw_control_triggers,
        "raw_control_trigger_rate_full_pool": rate(raw_control_triggers, n),
        "certified_negative_controls": certified_negative,
        "certified_negative_coverage": rate(certified_negative, n),
        "certified_false_triggers": certified_false_triggers,
        "certified_false_trigger_rate_full_pool": rate(
            certified_false_triggers, certified_negative
        ),
        "certified_false_trigger_rate_among_compatible": rate(
            certified_false_triggers, control_compatible
        ),
        "raw_contract_positive_controls": contract_positive,
        "raw_ambiguous_controls": ambiguous_controls,
        "triggered_faults": triggers,
        "witness_records": witness_records,
        "witness_valid": witnesses_valid,
        "witness_valid_rate_among_triggered": rate(witnesses_valid, triggers),
        "fault_status_counts": dict(sorted(Counter(row["fault_status"] for row in selected).items())),
        "control_status_counts": dict(
            sorted(Counter(row["certified_control_status"] for row in selected).items())
        ),
        "raw_control_status_counts": dict(
            sorted(Counter(row["raw_control_status"] for row in selected).items())
        ),
        "raw_control_certification_counts": dict(
            sorted(Counter(row["raw_control_certification"] for row in selected).items())
        ),
        "witness_status_counts": dict(sorted(Counter(row["reconstruction_status"] for row in selected).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report denominator-complete V2.2 executor replay metrics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fault-pool-manifest", required=True)
    parser.add_argument("--bird-fault-results", required=True)
    parser.add_argument("--spider-fault-results", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--control-certification", required=True)
    parser.add_argument("--certified-control-pool", required=True)
    parser.add_argument("--witness-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    fault_pool_manifest = json.loads((ROOT / args.fault_pool_manifest).read_text(encoding="utf-8"))
    fault_rows = []
    for dataset, path in (("bird", args.bird_fault_results), ("spider", args.spider_fault_results)):
        for row in read_jsonl(ROOT / path):
            fault_rows.append(
                {
                    "replay_key": replay_key(row),
                    "dataset": dataset,
                    "contract_type": row["contract_type"],
                    "fault_status": str((row.get("grain_result") or {}).get("status") or "ERROR"),
                    "evidence_scope_policy": row.get("evidence_scope_policy", "unspecified"),
                }
            )
    controls = {replay_key(row): row for row in read_jsonl(ROOT / args.controls)}
    certifications = {row["replay_key"]: row for row in read_tsv(ROOT / args.control_certification)}
    certified_controls = {
        replay_key(row): row for row in read_jsonl(ROOT / args.certified_control_pool)
    }
    witnesses = {replay_key(row): row for row in read_jsonl(ROOT / args.witness_audit)}
    if (
        len(controls) != len(fault_rows)
        or len(witnesses) != len(fault_rows)
        or len(certifications) != len(fault_rows)
        or len(certified_controls) != len(fault_rows)
    ):
        raise ValueError(
            "every emitted fault must have exactly one raw control, certification, certified control, and witness-audit record"
        )
    rows: list[dict[str, Any]] = []
    for fault in fault_rows:
        key = fault["replay_key"]
        if (
            key not in controls
            or key not in witnesses
            or key not in certifications
            or key not in certified_controls
        ):
            raise ValueError(
                f"missing raw control, certification, certified control, or witness audit for {key}"
            )
        control = controls[key]
        witness = witnesses[key]
        certification = certifications[key]
        certified_control = certified_controls[key]
        if control["dataset"] != fault["dataset"] or witness["dataset"] != fault["dataset"]:
            raise ValueError(f"dataset mismatch for {key}")
        rows.append(
            {
                **fault,
                "raw_control_status": control["control_status"],
                "raw_control_certification": certification["certification_label"],
                "certified_control_status": certified_control["control_status"],
                **witness,
            }
        )

    datasets = tuple(sorted({row["dataset"] for row in rows}))
    report_rows = [count_row(rows, dataset, operator) for dataset in datasets for operator in OPERATORS]
    report_rows.extend(count_row(rows, dataset, None) for dataset in datasets)
    report_rows.extend(count_row(rows, None, operator) for operator in OPERATORS)
    total = count_row(rows, None, None)
    report_rows.append(total)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "executor_validation_v2.2_unfiltered_controlled_fault_control_replay",
        "evidence_scope_policies": sorted({row["evidence_scope_policy"] for row in fault_rows}),
        "contract_freeze": manifest,
        "fault_pool_freeze": fault_pool_manifest,
        "selection_rule": {
            "fault_pool": "All 580 frozen Contract V2.1 accepted contracts are offered to deterministic broad mutation; no gold-PASS filter, detectable-only filter, or replay-result selection is permitted. A contract for which no deterministic mutation is emitted remains as FAULT_NOT_EMITTED in the pre-replay fault-construction coverage.",
            "controls": "Every frozen contract receives an independently constructed synthetic negative control before execution. Legacy reference controls are retained only as a raw diagnostic.",
        },
        "metric_definitions": {
            "executor_compatibility": "A replay side is compatible only when its status is PASS or VIOLATION; INVALID_WITNESS, UNSUPPORTED, ERROR, and missing outcomes remain in the full denominator.",
            "trigger_rate": "Fault VIOLATION rate, reported both over all emitted faults and among fault-executor-compatible rows.",
            "raw_control_trigger_rate": "Observed control VIOLATION rate over all controls; this is not a false-trigger estimate because reference controls may be intrinsically contract-positive.",
            "certified_false_trigger_rate": "Independent synthetic-control VIOLATIONs divided by all 580 controls certified negative from their matched construction without inspecting executor outcomes.",
            "witness_valid_rate": "Independently code-audited valid witnesses divided by all triggered faults; reconstruction failures count as invalid.",
        },
        "by_dataset_operator": report_rows,
        "overall": total,
        "reporting_guardrails": [
            "This is executor/validation evidence under controlled faults, not a new Contract V2.2 construction-quality claim.",
            "No V2.1 contract, reviewer label, RQ1 denominator, or paper-facing table is overwritten.",
            "Fault-construction losses are reported against all 580 frozen contracts; executor, trigger, control, and witness rates use the emitted-pair denominator stated in each row.",
            "Reference-SQL controls are retained only as a raw diagnostic. Specificity uses a separate full-census synthetic control pool certified safe by matched construction before execution.",
            "All compatibility losses, untriggered faults, control alerts, and witness failures remain visible in their declared denominators.",
        ],
    }
    (output_dir / "v22_executor_validation.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table_fields = [field for field in total if not field.endswith("_counts")]
    with (output_dir / "v22_executor_validation.table.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row[field] for field in table_fields} for row in report_rows)
    print(json.dumps(total, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
