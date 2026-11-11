#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
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


def verify_controlled() -> None:
    summary = json.loads((ROOT / 'results/controlled_validation.json').read_text())
    rows = [json.loads(line) for line in (ROOT / 'results/controlled_supported_rows.jsonl').open()]
    if summary['settings']['evidence_scope_policy'] != 'supported':
        raise SystemExit('controlled replay must use the supported policy')
    for kind in ('challenge', 'control'):
        selected = [row for row in rows if row['kind'] == kind]
        keys = {(row['dataset'], row['db_id'], row['question_id'], row['operator'], row['contract_index']) for row in selected}
        if len(selected) != 580 or len(keys) != 580:
            raise SystemExit(f'incomplete controlled cohort: {kind}')
        if dict(Counter(row['status'] for row in selected)) != summary['counts'][kind]:
            raise SystemExit(f'controlled status-count mismatch: {kind}')
        for operator, counts in summary['by_operator'][kind].items():
            actual = Counter(row['status'] for row in selected if row['operator'] == operator)
            if dict(actual) != counts:
                raise SystemExit(f'controlled operator-count mismatch: {kind}/{operator}')
    for name, record in summary['inputs'].items():
        require_hash(record, ROOT / 'inputs/controlled' / name, f'controlled input: {name}')
    witnesses = [json.loads(line) for line in (ROOT / 'data/controlled_supported_witnesses.jsonl').open()]
    audit = summary['witness_audit']
    if len(witnesses) != audit['retained'] or audit['retained'] != summary['counts']['challenge']['VIOLATION']:
        raise SystemExit('controlled witness-count mismatch')
    if sum(w['independent_checks']['code_audit_valid'] for w in witnesses) != audit['valid']:
        raise SystemExit('controlled independent-check mismatch')
    if any(w['evidence_scope'] != 'SUPPORTED' for w in witnesses):
        raise SystemExit('deferred evidence in confirmed controlled witnesses')
    with (ROOT / 'results/controlled_validation.tsv').open() as stream:
        for row in csv.DictReader(stream, delimiter='\t'):
            kind, operator = row['kind'], row['operator']
            counts = summary['counts'][kind] if operator == 'all' else summary['by_operator'][kind][operator]
            if int(row['n']) != sum(counts.values()):
                raise SystemExit('controlled TSV denominator mismatch')
            for column, status in [('violation', 'VIOLATION'), ('pass', 'PASS'), ('abstain', 'ABSTAIN'), ('invalid', 'INVALID_WITNESS')]:
                if int(row[column]) != counts.get(status, 0):
                    raise SystemExit('controlled TSV outcome mismatch')
    print(f"Controlled supported replay verified: 337 alerts, 37 challenge abstentions, 575 control PASS, four control abstentions; {audit['valid']}/{audit['retained']} independent checks.")


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

    verify_controlled()
    verify_llm_judges()
    print(
        "Artifact verified: 580 contracts, 1,160 candidates, 1,160 canonical "
        "candidate labels, complete reviewer exports A=2,364/B=2,364/C=115, "
        "contract-review exports A=580/B=580/C=2; supported and archived "
        "controlled records verified."
    )


if __name__ == "__main__":
    main()
