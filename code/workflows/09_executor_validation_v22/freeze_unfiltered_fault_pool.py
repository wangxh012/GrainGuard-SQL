"""Hash and validate the V2.2 fault pool before any executor replay occurs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key_from_fault(row: dict[str, Any]) -> tuple[Any, Any, str, int]:
    return (row["question_id"], row["db_id"], row["contract_type"], int(row.get("contract_index", 0)))


def key_from_manifest(row: dict[str, Any]) -> tuple[Any, Any, str, int]:
    return (row["question_id"], row["db_id"], row["contract_type"], int(row["contract_index"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the unfiltered V2.2 fault pool before replay.")
    parser.add_argument("--contract-manifest", required=True)
    parser.add_argument("--bird-faults", required=True)
    parser.add_argument("--spider-faults", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    contracts = json.loads((ROOT / args.contract_manifest).read_text(encoding="utf-8"))
    sources = {"bird": ROOT / args.bird_faults, "spider": ROOT / args.spider_faults}
    output: dict[str, Any] = {
        "protocol": "v22_unfiltered_fault_pool_pre_replay_freeze",
        "selection_prohibitions": [
            "no --require-gold-pass",
            "no --detectable-only",
            "no --prefer-validated-violation",
        ],
        "datasets": {},
    }
    for dataset, path in sources.items():
        rows = read_jsonl(path)
        expected = {key_from_manifest(row) for row in contracts["datasets"][dataset]["contract_keys"]}
        observed = {key_from_fault(row) for row in rows}
        if len(observed) != len(rows):
            raise ValueError(f"{dataset}: duplicate fault keys")
        extra = observed - expected
        if extra:
            raise ValueError(
                f"{dataset}: fault pool contains contracts outside the frozen V2.1 input; "
                f"extra={sorted(extra)[:3]}"
            )
        selected = [row for row in rows if row.get("generation_validation_status") is not None]
        if selected:
            raise ValueError(
                f"{dataset}: {len(selected)} faults carry a generation validation result; "
                "the pool may have been validation-selected"
            )
        output["datasets"][dataset] = {
            "faults": str(path),
            "faults_sha256": sha256(path),
            "frozen_contracts": len(expected),
            "emitted_faults": len(rows),
            "fault_construction_coverage": len(rows) / len(expected) if expected else None,
            "unemitted_contract_keys": [
                {
                    "question_id": key[0],
                    "db_id": key[1],
                    "contract_type": key[2],
                    "contract_index": key[3],
                    "pre_replay_status": "FAULT_NOT_EMITTED",
                }
                for key in sorted(expected - observed)
            ],
            "operator_counts": dict(sorted(Counter(row["contract_type"] for row in rows).items())),
            "mutation_source_counts": dict(sorted(Counter(row.get("source") or "unknown" for row in rows).items())),
        }
    output["emitted_faults_total"] = sum(item["emitted_faults"] for item in output["datasets"].values())
    output["frozen_contracts_total"] = contracts["accepted_contracts_total"]
    output["fault_construction_coverage"] = (
        output["emitted_faults_total"] / output["frozen_contracts_total"]
        if output["frozen_contracts_total"]
        else None
    )
    (ROOT / args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "fault_pool_manifest": args.out,
                "frozen_contracts_total": output["frozen_contracts_total"],
                "emitted_faults_total": output["emitted_faults_total"],
                "fault_construction_coverage": output["fault_construction_coverage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
