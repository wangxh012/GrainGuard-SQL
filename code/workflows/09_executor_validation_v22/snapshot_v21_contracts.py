"""Freeze the Contract V2.1 replay inputs inside an isolated V2.2 run.

This is deliberately a copy-and-hash operation.  It never writes the V2.1
source artifacts, review packets, or paper-facing tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ACCEPTED = {"bird": 322, "spider": 258}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Prefer repository-relative provenance but support external smoke outputs."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contract_key(row: dict[str, Any], index: int) -> dict[str, Any]:
    contracts = row.get("auto_contracts") or []
    contract = contracts[index]
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "question_id": row["question_id"],
        "db_id": row["db_id"],
        "contract_type": contract["contract_type"],
        "contract_index": index,
        "contract_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def inspect(dataset: str, path: Path, expected: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    accepted = [row for row in rows if row.get("status") == "accepted"]
    keys = [contract_key(row, index) for row in accepted for index in range(len(row.get("auto_contracts") or []))]
    if len(accepted) != expected or len(keys) != expected:
        raise ValueError(
            f"{dataset}: expected {expected} accepted one-contract V2.1 rows, found "
            f"{len(accepted)} rows / {len(keys)} contracts in {path}"
        )
    return {
        "source": display_path(path),
        "source_sha256": sha256(path),
        "rows": len(rows),
        "accepted_rows": len(accepted),
        "accepted_contracts": len(keys),
        "contract_keys": keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Contract V2.1 inputs for an executor-only V2.2 run.")
    parser.add_argument(
        "--bird-contracts",
        default="artifacts/runs/bird/shared/00_generation_inputs/contract_v21_recovery/auto_contracts.v21.candidate.jsonl",
    )
    parser.add_argument(
        "--spider-contracts",
        default="artifacts/runs/spider/shared/00_generation_inputs/contract_v21_recovery/auto_contracts.v21.candidate.jsonl",
    )
    parser.add_argument("--output-dir", default="artifacts/runs/v22_executor_validation")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    input_dir = output_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    sources = {"bird": ROOT / args.bird_contracts, "spider": ROOT / args.spider_contracts}
    manifest = {
        "protocol": "executor_validation_v2.2",
        "contract_policy": "Contract V2.1 is copied and hashed; schema, prompts, acceptance gate, review labels, and RQ1 denominators are not changed.",
        "expected_accepted_contracts": EXPECTED_ACCEPTED,
        "datasets": {},
    }
    for dataset, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        detail = inspect(dataset, source, EXPECTED_ACCEPTED[dataset])
        destination = input_dir / f"{dataset}.auto_contracts.v21.jsonl"
        shutil.copyfile(source, destination)
        detail["frozen_copy"] = display_path(destination)
        detail["frozen_copy_sha256"] = sha256(destination)
        manifest["datasets"][dataset] = detail
    manifest["accepted_contracts_total"] = sum(
        item["accepted_contracts"] for item in manifest["datasets"].values()
    )
    if manifest["accepted_contracts_total"] != 580:
        raise AssertionError("the V2.2 executor replay must freeze exactly 580 V2.1 contracts")
    (output_dir / "v21_contract_freeze.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "manifest": display_path(output_dir / "v21_contract_freeze.manifest.json"),
                "accepted_contracts_total": manifest["accepted_contracts_total"],
                "accepted_contracts_by_dataset": {
                    dataset: detail["accepted_contracts"]
                    for dataset, detail in manifest["datasets"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
