#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from verify_llm_judges import main as verify_llm_judges


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def row_count(path: Path) -> int:
    if path.suffix == ".tsv":
        with path.open(encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def require_hash(record: dict[str, Any], path: Path, context: str) -> None:
    if record.get("sha256") != digest(path):
        raise SystemExit(f"provenance hash mismatch: {context}")


def require_label_hashes(records: list[dict[str, Any]], context: str) -> None:
    expected = {
        "code/artifacts/annotations/sql_labels/final_reference_labels.paper_core.v2.jsonl": (
            ROOT
            / "code/artifacts/annotations/sql_labels/final_reference_labels.paper_core.v2.jsonl"
        ),
        "code/artifacts/annotations/sql_labels/supplemental_spec_failures.final.jsonl": (
            ROOT
            / "code/artifacts/annotations/sql_labels/supplemental_spec_failures.final.jsonl"
        ),
    }
    observed = {str(record.get("path")): record for record in records}
    if set(observed) != set(expected):
        raise SystemExit(f"unexpected label provenance paths: {context}")
    for relative, path in expected.items():
        require_hash(observed[relative], path, f"{context}: {relative}")


def require_contract_hashes(records: list[dict[str, Any]], context: str) -> None:
    expected = {
        reviewer: digest(ROOT / f"data/contract_annotator_{reviewer}_labels.jsonl")
        for reviewer in ("a", "b", "c")
    }
    for record in records:
        for reviewer, expected_hash in expected.items():
            if record.get(f"reviewer_{reviewer}_sha256") != expected_hash:
                raise SystemExit(
                    f"contract-review provenance mismatch: {context}: "
                    f"{record.get('dataset')}: reviewer {reviewer.upper()}"
                )


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["expected_rows"].items():
        actual = row_count(ROOT / "data" / name)
        if actual != expected:
            raise SystemExit(f"{name}: expected {expected} rows, found {actual}")
    for relative, metadata in manifest["files"].items():
        if digest(ROOT / relative) != metadata["sha256"]:
            raise SystemExit(f"checksum mismatch: {relative}")

    candidate_manifest_path = ROOT / "inputs/natural/v23_candidate_inputs.manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    require_label_hashes(candidate_manifest["label_sources"], "candidate manifest")
    for dataset, record in candidate_manifest["contracts"].items():
        require_hash(record, ROOT / record["path"], f"candidate contracts: {dataset}")
    for pool, record in candidate_manifest["outputs"].items():
        require_hash(
            record,
            ROOT / "inputs/natural" / f"{pool}_candidate.v23.jsonl",
            f"candidate output: {pool}",
        )

    natural_replay = json.loads(
        (ROOT / "results/natural_replay.json").read_text(encoding="utf-8")
    )
    require_label_hashes(natural_replay["label_sources"], "natural replay")
    for pool, record in natural_replay["inputs"].items():
        require_hash(record, ROOT / record["path"], f"natural replay input: {pool}")

    effectiveness = json.loads(
        (ROOT / "results/natural_effectiveness.json").read_text(encoding="utf-8")
    )
    require_label_hashes(effectiveness["inputs"]["labels"], "effectiveness")
    require_contract_hashes(
        effectiveness["inputs"]["contract_reviews"], "effectiveness"
    )
    require_hash(
        effectiveness["inputs"]["candidate_manifest"],
        candidate_manifest_path,
        "effectiveness candidate manifest",
    )

    label_audit = json.loads(
        (ROOT / "results/human_label_audit.json").read_text(encoding="utf-8")
    )
    audit_label_records = [
        label_audit["inputs"]["primary_labels"],
        label_audit["inputs"]["supplemental_labels"],
    ]
    require_label_hashes(audit_label_records, "human-label audit")
    require_contract_hashes(
        [
            {"dataset": dataset, **record}
            for dataset, record in label_audit["contract_reviews"].items()
        ],
        "human-label audit",
    )
    for name, record in label_audit["inputs"].items():
        if Path(record["path"]).is_absolute():
            raise SystemExit(f"absolute provenance path in human-label audit: {name}")

    verify_llm_judges()
    print(
        "Artifact verified: 580 contracts, 1,160 candidates, 1,160 canonical "
        "candidate labels, complete reviewer exports A=2,364/B=2,364/C=115, "
        "contract-review exports A=580/B=580/C=2, 365 witnesses, 579 controls."
    )


if __name__ == "__main__":
    main()
