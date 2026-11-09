#!/usr/bin/env python3
"""Audit frozen A/B/C decisions used by natural-candidate V2.3.

This program never changes an annotation.  It verifies the frozen decision
chains, joins them to the V2.3 replay, and creates a label/outcome ledger.  It
also emits an outcome-blinded packet for a separate target-alignment audit:
the SQL-level label asks whether a query contains any metric-grain defect,
whereas the contract-relative detector can test only the selected V2.1 target.
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

from contract_review_exports import read_review_labels, review_label_path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.evidence_scope import emitted_trace_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument(
        "--primary-labels",
        default="artifacts/annotations/sql_labels/final_reference_labels.paper_core.v2.jsonl",
    )
    parser.add_argument(
        "--primary-manifest",
        default="artifacts/annotations/sql_labels/final_reference_labels.paper_core.v2.manifest.json",
    )
    parser.add_argument(
        "--supplemental-labels",
        default="artifacts/annotations/sql_labels/supplemental_spec_failures.final.jsonl",
    )
    parser.add_argument(
        "--supplemental-manifest",
        default="artifacts/annotations/sql_labels/supplemental_spec_failures.final.manifest.json",
    )
    parser.add_argument(
        "--bird-contract-review-dir",
        default="../data/contract_review/bird",
    )
    parser.add_argument(
        "--spider-contract-review-dir",
        default="../data/contract_review/spider",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_model(value: Any) -> str:
    key = str(value or "").casefold()
    aliases = {
        "glm": "glm-5.1",
        "glm-5.1": "glm-5.1",
        "qwen": "qwen3.6-27b",
        "qwen3.6-27b": "qwen3.6-27b",
    }
    if key not in aliases:
        raise ValueError(f"unsupported model: {value!r}")
    return aliases[key]


def label_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["dataset"]).casefold(),
        canonical_model(row["model"]),
        int(row["question_id"]),
        str(row["db_id"]),
    )


def trace_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    dataset, db_id, question_id, _operator, _index = str(row["sample_id"]).rsplit(
        ":", 4
    )
    model = canonical_model(str(row["pool"]).rsplit("_", 1)[-1])
    return dataset, model, int(question_id), db_id


def trace_status(row: dict[str, Any]) -> str:
    return emitted_trace_status(row)


def verify_output_hash(manifest_path: Path, label_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["final_labels_sha256"]
    observed = sha256(label_path)
    if expected != observed:
        raise ValueError(f"frozen-label hash mismatch: {label_path}")
    return manifest


def validate_label_chain(row: dict[str, Any]) -> None:
    relation = row.get("a_b_relation")
    a = row.get("annotator_a_positive")
    b = row.get("annotator_b_positive")
    c = row.get("annotator_c_positive")
    final = bool(row["final_paper_core_positive"])
    if relation == "agree":
        if a is None or b is None or bool(a) != bool(b) or c is not None:
            raise ValueError(f"invalid A/B agreement chain: {row.get('sample_id')}")
        expected = bool(a)
    elif relation == "disagree":
        if a is None or b is None or bool(a) == bool(b) or c is None:
            raise ValueError(f"invalid A/B disagreement chain: {row.get('sample_id')}")
        expected = bool(c)
    elif relation == "b_abstained":
        if a is None or b is not None or c is None:
            raise ValueError(f"invalid B-abstention chain: {row.get('sample_id')}")
        expected = bool(c)
    else:
        raise ValueError(f"unknown A/B relation: {relation!r}")
    if final != expected:
        raise ValueError(f"final label does not follow decision chain: {row.get('sample_id')}")


def load_labels(
    primary_path: Path, supplemental_path: Path
) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    result: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for source, path in (("primary", primary_path), ("supplemental", supplemental_path)):
        for row in read_jsonl(path):
            if row.get("sql_role") != "candidate_sql":
                continue
            validate_label_chain(row)
            key = label_key(row)
            if key in result:
                raise ValueError(f"duplicate candidate label: {key}")
            result[key] = {**row, "_label_source": source}
    return result


def audit_contract_reviews(
    dataset: str, directory: Path
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[str, Any]]:
    review_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    packets = {
        (int(row["question_id"]), str(row["db_id"])): row
        for row in read_jsonl(directory / "reviewer_a.packet.jsonl")
    }
    streams = {
        reviewer: {
            (int(row["question_id"]), str(row["db_id"])): row
            for row in read_review_labels(directory, reviewer, dataset)
        }
        for reviewer in ("a", "b")
    }
    if set(packets) != set(streams["a"]) or set(packets) != set(streams["b"]):
        raise ValueError(f"{dataset}: incomplete contract-review coverage")
    disagreements = {
        key
        for key in packets
        if streams["a"][key]["decision"] != streams["b"][key]["decision"]
    }
    adjudications = {
        (int(row["question_id"]), str(row["db_id"])): row
        for row in read_review_labels(directory, "c", dataset)
    }
    if set(adjudications) != disagreements:
        raise ValueError(f"{dataset}: C must adjudicate exactly the A/B disagreements")

    final: dict[tuple[int, str], dict[str, Any]] = {}
    for key in packets:
        a, b = streams["a"][key], streams["b"][key]
        chosen = a if key not in disagreements else adjudications[key]
        for reviewer, row in (("A", a), ("B", b)):
            decision = str(row["decision"])
            adequate = bool(row["operator_adequate"])
            if (decision == "ACCEPT") != adequate:
                raise ValueError(
                    f"{dataset} {key}: {reviewer} decision/operator_adequate mismatch"
                )
        final[key] = {
            "decision": chosen["decision"],
            "a_decision": a["decision"],
            "b_decision": b["decision"],
            "c_decision": adjudications.get(key, {}).get("decision"),
            "packet": packets[key],
        }

    identical_rationales = sum(
        streams["a"][key].get("rationale") == streams["b"][key].get("rationale")
        for key in packets
    )
    summary = {
        "rows": len(packets),
        "a_decisions": dict(Counter(row["decision"] for row in streams["a"].values())),
        "b_decisions": dict(Counter(row["decision"] for row in streams["b"].values())),
        "a_b_exact_decision_agreement": len(packets) - len(disagreements),
        "a_b_disagreements": len(disagreements),
        "c_adjudications": len(adjudications),
        "final_accept": sum(row["decision"] == "ACCEPT" for row in final.values()),
        "identical_a_b_rationales": identical_rationales,
        "rationale_export_note": review_manifest.get("rationale_export_note"),
        "gold_sql_visibility_scope": review_manifest.get("gold_sql_visibility_scope"),
        "system_reference_sql_access": review_manifest.get("system_reference_sql_access"),
        "reviewer_a_sha256": sha256(review_label_path(directory, "a")),
        "reviewer_b_sha256": sha256(review_label_path(directory, "b")),
        "reviewer_c_sha256": sha256(review_label_path(directory, "c")),
    }
    return final, summary


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def main() -> None:
    args = parse_args()
    trace_path = resolve(args.traces)
    primary_path = resolve(args.primary_labels)
    supplemental_path = resolve(args.supplemental_labels)
    primary_manifest = verify_output_hash(resolve(args.primary_manifest), primary_path)
    supplemental_manifest = verify_output_hash(
        resolve(args.supplemental_manifest), supplemental_path
    )
    labels = load_labels(primary_path, supplemental_path)

    contract_reviews: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    contract_summary: dict[str, Any] = {}
    for dataset, directory in (
        ("bird", resolve(args.bird_contract_review_dir)),
        ("spider", resolve(args.spider_contract_review_dir)),
    ):
        contract_reviews[dataset], contract_summary[dataset] = audit_contract_reviews(
            dataset, directory
        )

    traces = read_jsonl(trace_path)
    if len(traces) != 1160:
        raise ValueError(f"expected 1,160 V2.3 traces, found {len(traces)}")
    ledger: list[dict[str, Any]] = []
    audit_packet: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    for trace in traces:
        key = trace_key(trace)
        if key in seen or key not in labels:
            raise ValueError(f"duplicate or unlabeled V2.3 trace: {key}")
        seen.add(key)
        label = labels[key]
        dataset, model, question_id, db_id = key
        contract_review = contract_reviews[dataset][(question_id, db_id)]
        status = trace_status(trace)
        prediction = status == "VIOLATION"
        truth = bool(label["final_paper_core_positive"])
        outcome = (
            "TP" if truth and prediction else
            "FN" if truth else
            "FP" if prediction else
            "TN"
        )
        operator = str(trace["sample_id"]).rsplit(":", 2)[-2]
        ledger.append(
            {
                "dataset": dataset,
                "model": model,
                "question_id": question_id,
                "db_id": db_id,
                "operator": operator,
                "status": status,
                "outcome": outcome,
                "final_positive": truth,
                "label_source": label["_label_source"],
                "a_positive": label.get("annotator_a_positive"),
                "b_positive": label.get("annotator_b_positive"),
                "c_positive": label.get("annotator_c_positive"),
                "a_b_relation": label.get("a_b_relation"),
                "resolution_basis": label.get("resolution_basis"),
                "contract_review_final": contract_review["decision"],
                "contract_review_a": contract_review["a_decision"],
                "contract_review_b": contract_review["b_decision"],
                "contract_review_c": contract_review["c_decision"],
                "sample_id": label.get("sample_id"),
            }
        )

        # Audit every positive (the recall denominator), every alarm (the
        # precision denominator), every item whose SQL label required C, and a
        # deterministic 10% audit of the remaining double-negative controls.
        # No executor result, prior label, or selection reason is exposed in
        # the packet itself.
        audit_selector = int(
            hashlib.sha256(
                f"v23-control-audit:{dataset}:{model}:{question_id}:{db_id}".encode()
            ).hexdigest()[:8],
            16,
        ) % 10
        if truth or prediction or label.get("a_b_relation") != "agree" or audit_selector == 0:
            packet = contract_review["packet"]
            packet_id = hashlib.sha256(
                f"v23-target-audit:{dataset}:{model}:{question_id}:{db_id}".encode()
            ).hexdigest()[:16]
            audit_packet.append(
                {
                    "packet_id": f"TA{packet_id}",
                    "dataset": dataset,
                    "model": model,
                    "question_id": question_id,
                    "db_id": db_id,
                    "question": label.get("question") or packet.get("question") or "",
                    "evidence": packet.get("evidence") or "",
                    "schema_registry_path": packet.get("schema_registry_path"),
                    "candidate_sql": label.get("candidate_sql") or label.get("sql"),
                    "contract_v21": packet.get("contract_v21"),
                    "review_instruction": (
                        "Assign CONTRACT_TARGET_VIOLATION when the SQL violates the "
                        "selected Contract V2.1 target; OUTSIDE_TARGET_DEFECT_ONLY when "
                        "it has a metric-grain defect that does not violate that target; "
                        "NO_METRIC_GRAIN_DEFECT when neither applies; or UNCERTAIN. Do "
                        "not inspect A/B/C labels or GrainGuard-SQL results."
                    ),
                }
            )

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_fields = [
        "dataset", "model", "question_id", "db_id", "operator", "status",
        "outcome", "final_positive", "label_source", "a_positive", "b_positive",
        "c_positive", "a_b_relation", "resolution_basis", "contract_review_final",
        "contract_review_a", "contract_review_b", "contract_review_c", "sample_id",
    ]
    write_tsv(output_dir / "v23_abc_outcome_ledger.tsv", ledger, ledger_fields)
    with (output_dir / "v23_target_alignment_audit.packet.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in sorted(audit_packet, key=lambda item: item["packet_id"]):
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    template_fields = ["packet_id", "target_relative_decision", "rationale"]
    write_tsv(
        output_dir / "v23_target_alignment_audit.blank.tsv",
        [{"packet_id": row["packet_id"]} for row in sorted(audit_packet, key=lambda item: item["packet_id"])],
        template_fields,
    )

    summary = {
        "protocol": "v23_frozen_abc_decision_chain_audit_v1",
        "mutation_policy": "read_only_frozen_labels_no_decision_changed",
        "candidate_rows": len(ledger),
        "outcomes": dict(Counter(row["outcome"] for row in ledger)),
        "selected_label_sources": dict(Counter(row["label_source"] for row in ledger)),
        "selected_a_b_relations": dict(Counter(row["a_b_relation"] for row in ledger)),
        "positive_resolution": dict(
            Counter(
                f"{row['label_source']}:{row['a_b_relation']}:{row['resolution_basis']}"
                for row in ledger if row["final_positive"]
            )
        ),
        "contract_reviews": contract_summary,
        "contract_final_accept": sum(
            row["final_accept"] for row in contract_summary.values()
        ),
        "target_alignment_audit_rows": len(audit_packet),
        "target_alignment_estimand_scope": (
            "The packet exhaustively covers frozen SQL positives, V2.3 alarms, "
            "and A/B disagreements or abstentions, plus a deterministic 10% audit "
            "of remaining double negatives. It can adjudicate alarm precision and "
            "the target relation of known SQL-level positives. It does not by itself "
            "provide a full-census target-relative recall denominator unless the "
            "unaudited double negatives are assumed not to contain label false negatives."
        ),
        "provenance_audit": {
            "primary_reference_status": primary_manifest.get("reference_status"),
            "supplemental_reference_status": supplemental_manifest.get("reference_status"),
            "supplemental_note": supplemental_manifest.get("provenance_note"),
        },
        "inputs": {
            "traces": {"path": str(trace_path), "sha256": sha256(trace_path)},
            "primary_labels": {"path": str(primary_path), "sha256": sha256(primary_path)},
            "supplemental_labels": {
                "path": str(supplemental_path), "sha256": sha256(supplemental_path)
            },
        },
    }
    summary_path = output_dir / "v23_abc_decision_audit.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
