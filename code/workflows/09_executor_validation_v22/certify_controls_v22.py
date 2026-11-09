"""Assign outcome-independent certification labels to V2.2 controls.

Certification is based only on the predeclared construction strategy.  The
executor outcome is copied for reporting but is never used to choose a label.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CERTIFIED_SAFE_STRATEGIES = {
    "neutralized_synthetic_trigger",
    "restore_contract_sum_on_same_multihop_path",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def certify(row: dict[str, Any]) -> tuple[str, str, str]:
    strategy = str(row.get("control_strategy") or "")
    if strategy in CERTIFIED_SAFE_STRATEGIES:
        return (
            "CERTIFIED_NEGATIVE",
            "SAFE_BY_MATCHED_CONSTRUCTION",
            "The trigger term or wrong ranking aggregate is neutralized on the same query skeleton/path; certification does not inspect replay status.",
        )
    if strategy in {"pre_injection_gold_sql", "reference_gold_sql_control"}:
        return (
            "AMBIGUOUS",
            "REFERENCE_SQL_NOT_NEGATIVE_CERTIFIED",
            "Reference SQL may contain an intrinsic contract violation; V2.1 contract-adequacy review does not certify it as a negative control.",
        )
    return (
        "AMBIGUOUS",
        "UNRECOGNIZED_CONTROL_CONSTRUCTION",
        "No outcome-independent negative-control proof is registered for this construction strategy.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify V2.2 controls without using executor outcomes.")
    parser.add_argument("--controls", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    controls = read_jsonl(ROOT / args.controls)
    rows: list[dict[str, Any]] = []
    for control in controls:
        label, basis, note = certify(control)
        rows.append(
            {
                "replay_key": control["replay_key"],
                "dataset": control["dataset"],
                "question_id": control["question_id"],
                "db_id": control["db_id"],
                "contract_type": control["contract_type"],
                "contract_index": int(control.get("contract_index", 0)),
                "control_strategy": control.get("control_strategy") or "",
                "certification_label": label,
                "certification_basis": basis,
                "certification_note": note,
                "observed_control_status": control.get("control_status") or "ERROR",
            }
        )
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "controls": len(rows),
                "certification_labels": dict(sorted(Counter(row["certification_label"] for row in rows).items())),
                "certified_negative_alerts": sum(
                    row["certification_label"] == "CERTIFIED_NEGATIVE"
                    and row["observed_control_status"] == "VIOLATION"
                    for row in rows
                ),
                "out": args.out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
