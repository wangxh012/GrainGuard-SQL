"""Classify V2.2 INVALID_WITNESS rows without changing replay outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict[str, Any]) -> tuple[Any, Any, str, int]:
    return (
        row["question_id"],
        row["db_id"],
        row["contract_type"],
        int(row.get("contract_index", 0)),
    )


def category(diagnosis: str) -> str:
    if "did not preserve contract value" in diagnosis:
        return "CONTRACT_NOT_PRESERVED"
    if "ambiguous column name" in diagnosis:
        return "AMBIGUOUS_COLUMN"
    if "no such column" in diagnosis:
        return "OUT_OF_SCOPE_COLUMN_REFERENCE"
    if "directed child-to-parent join path" in diagnosis:
        return "UNSUPPORTED_TST_PATH_DIRECTION"
    return "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage V2.2 invalid witnesses.")
    parser.add_argument("--dataset-config-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(args.dataset_config_json)
    details: list[dict[str, Any]] = []
    for dataset, paths in sorted(config.items()):
        faults = {key(row): row for row in read_jsonl(ROOT / paths["faults"])}
        for result in read_jsonl(ROOT / paths["results"]):
            if (result.get("grain_result") or {}).get("status") != "INVALID_WITNESS":
                continue
            fault = faults[key(result)]
            diagnoses = [
                str(item.get("diagnosis") or "")
                for item in result["grain_result"].get("results", [])
                if item.get("status") == "INVALID_WITNESS"
            ]
            categories = sorted({category(item) for item in diagnoses})
            details.append(
                {
                    "dataset": dataset,
                    "question_id": result["question_id"],
                    "db_id": result["db_id"],
                    "contract_type": result["contract_type"],
                    "contract_index": int(result.get("contract_index", 0)),
                    "fault_source": fault.get("source"),
                    "categories": categories,
                    "diagnoses": diagnoses,
                }
            )
    by_operator = Counter(row["contract_type"] for row in details)
    by_category = Counter(category for row in details for category in row["categories"])
    summary = {
        "protocol": "v22_invalid_witness_post_replay_triage",
        "invalid_witness_rows": len(details),
        "by_operator": dict(sorted(by_operator.items())),
        "by_category": dict(sorted(by_category.items())),
        "decision": (
            "Retain these rows as executor-compatibility losses. Do not tune the frozen "
            "fault pool or PairBuilder to individual question IDs; a future generic "
            "scope-aware SQL injector or path-general witness revision requires a new "
            "executor replay, not a Contract V2.1 rerun."
        ),
        "rows": details,
    }
    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("invalid_witness_rows", "by_operator", "by_category")}, indent=2))


if __name__ == "__main__":
    main()
