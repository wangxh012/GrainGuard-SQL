#!/usr/bin/env python3
"""Summarize Contract V2.1 natural-candidate replay on conditional and full pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from collections import Counter
from pathlib import Path
from typing import Any

from contract_review_exports import read_review_labels, review_label_path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.evidence_scope import (  # noqa: E402
    SUPPORTED_POLICY,
    UNRESTRICTED_POLICY,
    emitted_trace_status,
    recorded_trace_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument(
        "--primary-labels",
        default="artifacts/annotations/sql_labels/final_reference_labels.paper_core.v2.jsonl",
    )
    parser.add_argument(
        "--supplemental-labels",
        default="artifacts/annotations/sql_labels/supplemental_spec_failures.final.jsonl",
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
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    """Validate the packaged candidate files against the frozen manifest.

    The manifest retains source-repository paths for provenance.  An artifact
    copy is portable, so validation uses the colocated files and their hashes
    rather than requiring those historical paths to exist on the reader's
    machine.
    """
    expected_pools = ("bird_glm", "bird_qwen", "spider_glm", "spider_qwen")
    outputs = manifest.get("outputs") or {}
    if set(outputs) != set(expected_pools):
        raise ValueError("candidate manifest does not contain the four closed V2.3 pools")
    for pool in expected_pools:
        expected_path = manifest_path.parent / f"{pool}_candidate.v23.jsonl"
        if not expected_path.is_file():
            raise ValueError(f"{pool}: manifest output is missing: {expected_path}")
        if outputs[pool].get("sha256") != sha256(expected_path):
            raise ValueError(f"{pool}: manifest output hash does not match the candidate file")


def canonical_model(value: Any) -> str:
    aliases = {
        "glm": "glm-5.1",
        "glm-5.1": "glm-5.1",
        "qwen": "qwen3.6-27b",
        "qwen3.6-27b": "qwen3.6-27b",
    }
    key = str(value or "").casefold()
    if key not in aliases:
        raise ValueError(f"unsupported model: {value!r}")
    return aliases[key]


def trace_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    dataset, db_id, question_id, _operator, _index = str(row["sample_id"]).rsplit(":", 4)
    model = canonical_model(str(row["pool"]).rsplit("_", 1)[-1])
    return dataset, model, int(question_id), db_id


def trace_operator(row: dict[str, Any]) -> str:
    return str(row["sample_id"]).rsplit(":", 2)[-2]


def trace_status(
    row: dict[str, Any],
    evidence_scope_policy: str = SUPPORTED_POLICY,
) -> str:
    if evidence_scope_policy == SUPPORTED_POLICY:
        return emitted_trace_status(row, evidence_scope_policy)
    return recorded_trace_status(row, evidence_scope_policy)


def adequate_contract_keys(review_dir: Path, dataset: str) -> set[tuple[str, int, str]]:
    streams = {}
    for reviewer in ("a", "b"):
        streams[reviewer] = {
            (int(row["question_id"]), str(row["db_id"])): str(row["decision"])
            for row in read_review_labels(review_dir, reviewer, dataset)
        }
    if set(streams["a"]) != set(streams["b"]):
        raise ValueError(f"{dataset}: A/B contract-review keys do not match")
    adjudications = {
        (int(row["question_id"]), str(row["db_id"])): str(row["decision"])
        for row in read_review_labels(review_dir, "c", dataset)
    }
    accepted: set[tuple[str, int, str]] = set()
    for key in streams["a"]:
        a, b = streams["a"][key], streams["b"][key]
        decision = a if a == b else adjudications.get(key)
        if decision is None:
            raise ValueError(f"{dataset}: missing C adjudication for {key}")
        if decision == "ACCEPT":
            accepted.add((dataset, key[0], key[1]))
    return accepted


def classification_metrics(
    labels: list[bool], predictions: list[bool]
) -> dict[str, float | None]:
    tp = sum(label and prediction for label, prediction in zip(labels, predictions))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    denominator = 2 * tp + fp + fn
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / denominator if denominator else None,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def hierarchical_indices(
    db_ids: list[str], question_ids: list[int], rng: random.Random
) -> list[int]:
    by_database: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (db_id, question_id) in enumerate(zip(db_ids, question_ids)):
        by_database[db_id][question_id].append(index)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for db_id in sorted(by_database):
        by_dataset[db_id.split(":", 1)[0]].append(db_id)
    sampled: list[int] = []
    for dataset in sorted(by_dataset):
        databases = by_dataset[dataset]
        for _ in databases:
            selected_db = rng.choice(databases)
            questions = by_database[selected_db]
            available = sorted(questions)
            for _ in available:
                sampled.extend(questions[rng.choice(available)])
    return sampled


def cluster_bootstrap_metrics(
    labels: list[bool],
    predictions: list[bool],
    db_ids: list[str],
    question_ids: list[int],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    observed = classification_metrics(labels, predictions)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {name: [] for name in ("precision", "recall", "f1")}
    undefined = {name: 0 for name in draws}
    for _ in range(samples):
        indices = hierarchical_indices(db_ids, question_ids, rng)
        result = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
        for name in draws:
            value = result[name]
            if value is None:
                undefined[name] += 1
            else:
                draws[name].append(float(value))
    return {
        name: {
            "estimate": observed[name],
            "ci_95": [quantile(values, 0.025), quantile(values, 0.975)] if values else None,
            "defined_bootstrap_replicates": len(values),
            "undefined_bootstrap_replicates": undefined[name],
        }
        for name, values in draws.items()
    }


def database_macro_metrics(
    labels: list[bool], predictions: list[bool], db_ids: list[str]
) -> dict[str, Any]:
    per_database: dict[str, dict[str, float | None]] = {}
    for db_id in sorted(set(db_ids)):
        indices = [index for index, value in enumerate(db_ids) if value == db_id]
        per_database[db_id] = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
    result: dict[str, Any] = {"database_count": len(per_database), "per_database": per_database}
    for name in ("precision", "recall", "f1"):
        values = [
            float(row[name]) for row in per_database.values() if row[name] is not None
        ]
        result[name] = sum(values) / len(values) if values else None
        result[f"{name}_eligible_databases"] = len(values)
    return result


def leave_one_database_out_metrics(
    labels: list[bool], predictions: list[bool], db_ids: list[str]
) -> dict[str, Any]:
    observed = classification_metrics(labels, predictions)
    estimates: dict[str, dict[str, float | None]] = {}
    for omitted in sorted(set(db_ids)):
        indices = [index for index, value in enumerate(db_ids) if value != omitted]
        estimates[omitted] = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
        )
    summary: dict[str, Any] = {"per_omitted_database": estimates}
    for name in ("precision", "recall", "f1"):
        values = [float(row[name]) for row in estimates.values() if row[name] is not None]
        point = observed[name]
        summary[name] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "max_absolute_change": (
                max(abs(value - float(point)) for value in values)
                if values and point is not None
                else None
            ),
        }
    return summary


def metrics(records: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    labels = [bool(row["label"]) for row in records]
    predictions = [bool(row["prediction"]) for row in records]
    db_ids = [f"{row['dataset']}:{row['db_id']}" for row in records]
    question_ids = [int(row["question_id"]) for row in records]
    point = classification_metrics(labels, predictions)
    decided = sum(row["status"] in {"PASS", "VIOLATION"} for row in records)
    result = {
        "n": len(records),
        "positive": sum(labels),
        "negative": len(records) - sum(labels),
        "tp": int(point["tp"]),
        "fp": int(point["fp"]),
        "fn": int(point["fn"]),
        "tn": len(records) - int(point["tp"] + point["fp"] + point["fn"]),
        "precision": point["precision"],
        "recall": point["recall"],
        "f1": point["f1"],
        "decided": decided,
        "decision_coverage": decided / len(records),
        "status_counts": dict(sorted(Counter(row["status"] for row in records).items())),
    }
    if samples:
        result["cluster_bootstrap"] = cluster_bootstrap_metrics(
            labels, predictions, db_ids, question_ids, samples=samples, seed=seed
        )
        result["database_macro"] = database_macro_metrics(labels, predictions, db_ids)
        result["leave_one_database_out"] = leave_one_database_out_metrics(
            labels, predictions, db_ids
        )
    return result


def pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.2f}"


def main() -> None:
    args = parse_args()
    trace_path = resolve(args.traces)
    manifest_path = resolve(args.input_manifest)
    output_dir = resolve(args.output_dir)
    label_paths = [resolve(args.primary_labels), resolve(args.supplemental_labels)]
    review_dirs = {
        "bird": resolve(args.bird_contract_review_dir),
        "spider": resolve(args.spider_contract_review_dir),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("accepted_questions") != 580 or manifest.get("candidate_rows") != 1160:
        raise ValueError("input manifest is not the closed 580-question/1,160-candidate V2.3 pool")
    validate_candidate_manifest(manifest, manifest_path)

    traces = read_jsonl(trace_path)
    if len(traces) != 1160:
        raise ValueError(f"expected 1,160 replay traces, found {len(traces)}")
    trace_by_key = {trace_key(row): row for row in traces}
    if len(trace_by_key) != len(traces):
        raise ValueError("duplicate V2.3 replay trace key")

    labels: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for path in label_paths:
        for row in read_jsonl(path):
            if row.get("sql_role") != "candidate_sql":
                continue
            key = (
                str(row["dataset"]).casefold(),
                canonical_model(row["model"]),
                int(row["question_id"]),
                str(row["db_id"]),
            )
            if key in labels:
                raise ValueError(f"duplicate label key: {key}")
            labels[key] = row
    if len(labels) != 1634:
        raise ValueError(f"expected 1,634 frozen candidate labels, found {len(labels)}")
    if set(trace_by_key) - set(labels):
        raise ValueError("replay contains candidates absent from the frozen label census")

    full_records: list[dict[str, Any]] = []
    for key, label_row in sorted(labels.items()):
        dataset, model, question_id, db_id = key
        trace = trace_by_key.get(key)
        status = trace_status(trace) if trace else "SPECIFICATION_ABSTENTION"
        full_records.append(
            {
                "dataset": dataset,
                "model": model,
                "question_id": question_id,
                "db_id": db_id,
                "gate_valid": trace is not None,
                "label": bool(label_row["final_paper_core_positive"]),
                "prediction": status == "VIOLATION",
                "status": status,
            }
        )
    gate_records = [row for row in full_records if row["gate_valid"]]
    if len(gate_records) != 1160:
        raise ValueError(f"gate-valid record closure failed: {len(gate_records)}")
    adequate_keys = set().union(
        *(adequate_contract_keys(path, dataset) for dataset, path in review_dirs.items())
    )
    if len(adequate_keys) != 573:
        raise ValueError(f"expected 573 adequate V2.1 contracts, found {len(adequate_keys)}")
    adequate_records = [
        row
        for row in gate_records
        if (row["dataset"], row["question_id"], row["db_id"]) in adequate_keys
    ]
    if len(adequate_records) != 1146:
        raise ValueError(
            f"adequate-contract candidate closure failed: expected 1,146, found {len(adequate_records)}"
        )

    rows: list[dict[str, Any]] = []
    pools = (
        ("bird", "glm-5.1"),
        ("bird", "qwen3.6-27b"),
        ("spider", "glm-5.1"),
        ("spider", "qwen3.6-27b"),
    )
    for scope, records in (
        ("full_pool", full_records),
        ("gate_valid", gate_records),
        ("adequate_contract", adequate_records),
    ):
        for offset, (dataset, model) in enumerate(pools):
            selected = [
                row for row in records if row["dataset"] == dataset and row["model"] == model
            ]
            rows.append(
                {
                    "scope": scope,
                    "dataset": dataset,
                    "model": model,
                    **metrics(selected, args.bootstrap_samples, args.seed + offset),
                }
            )
        rows.append(
            {
                "scope": scope,
                "dataset": "all",
                "model": "all",
                **metrics(records, args.bootstrap_samples, args.seed + 100),
            }
        )

    gate_selection = []
    for dataset in ("bird", "spider", "all"):
        selected = [
            row for row in full_records if dataset == "all" or row["dataset"] == dataset
        ]
        accepted = [row for row in selected if row["gate_valid"]]
        rejected = [row for row in selected if not row["gate_valid"]]
        accepted_positive = sum(bool(row["label"]) for row in accepted)
        rejected_positive = sum(bool(row["label"]) for row in rejected)
        accepted_rate = accepted_positive / len(accepted) if accepted else None
        rejected_rate = rejected_positive / len(rejected) if rejected else None
        gate_selection.append(
            {
                "dataset": dataset,
                "accepted_candidates": len(accepted),
                "accepted_positive": accepted_positive,
                "accepted_positive_rate": accepted_rate,
                "rejected_candidates": len(rejected),
                "rejected_positive": rejected_positive,
                "rejected_positive_rate": rejected_rate,
                "rejected_to_accepted_prevalence_ratio": (
                    rejected_rate / accepted_rate
                    if accepted_rate and rejected_rate is not None
                    else None
                ),
            }
        )

    operator_rows: list[dict[str, Any]] = []
    for operator in sorted({trace_operator(row) for row in traces}):
        keys = {key for key, row in trace_by_key.items() if trace_operator(row) == operator}
        selected = [
            row
            for row in gate_records
            if (row["dataset"], row["model"], row["question_id"], row["db_id"]) in keys
        ]
        operator_rows.append(
            {
                "operator": operator,
                **metrics(
                    [
                        {
                            **row,
                            "status": trace_status(
                                trace_by_key[
                                    (
                                        row["dataset"],
                                        row["model"],
                                        row["question_id"],
                                        row["db_id"],
                                    )
                                ],
                                UNRESTRICTED_POLICY,
                            ),
                            "prediction": trace_status(
                                trace_by_key[
                                    (
                                        row["dataset"],
                                        row["model"],
                                        row["question_id"],
                                        row["db_id"],
                                    )
                                ],
                                UNRESTRICTED_POLICY,
                            )
                            == "VIOLATION",
                        }
                        for row in selected
                    ],
                    0,
                    args.seed,
                ),
            }
        )

    report = {
        "experiment": "natural_candidate_effectiveness_v2.3_contract_v2.1",
        "reference_estimand": (
            "End-to-end detection of any frozen SQL-level metric-grain defect. "
            "A false negative can therefore arise from contract target selection, "
            "executor coverage, or witness search; it is not an executor-only miss."
        ),
        "target_relative_estimand_status": (
            "Not estimated until the outcome-blinded Contract V2.1 target-alignment "
            "audit is completed."
        ),
        "primary_estimand": (
            "Full 1,634-candidate pool; Contract V2.1 construction failures and "
            "post-contract unsupported outcomes are retained as abstentions."
        ),
        "conditional_estimand": "The 1,160 candidates attached to 580 gate-valid Contract V2.1 questions.",
        "inputs": {
            "trace": {"path": display_path(trace_path), "sha256": sha256(trace_path)},
            "candidate_manifest": {
                "path": manifest_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(manifest_path),
            },
            "labels": [
                {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
                for path in label_paths
            ],
            "contract_reviews": [
                {
                    "dataset": dataset,
                    "directory": path.relative_to(ROOT).as_posix(),
                    "reviewer_a_sha256": sha256(review_label_path(path, "a")),
                    "reviewer_b_sha256": sha256(review_label_path(path, "b")),
                    "reviewer_c_sha256": sha256(review_label_path(path, "c")),
                }
                for dataset, path in review_dirs.items()
            ],
        },
        "rows": rows,
        "gate_selection": gate_selection,
        "gate_valid_by_operator": operator_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "v23_effectiveness.summary.json"
    summary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table_path = output_dir / "v23_effectiveness.table.tsv"
    columns = [
        "scope", "dataset", "model", "n", "positive", "tp", "fp", "fn", "tn",
        "precision_pct", "recall_pct", "f1_pct", "decided", "decision_coverage_pct",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "scope": row["scope"],
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "n": row["n"],
                    "positive": row["positive"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "tn": row["tn"],
                    "precision_pct": pct(row["precision"]),
                    "recall_pct": pct(row["recall"]),
                    "f1_pct": pct(row["f1"]),
                    "decided": row["decided"],
                    "decision_coverage_pct": pct(row["decision_coverage"]),
                }
            )
    print(json.dumps({"summary": str(summary_path), "table": str(table_path)}, indent=2))


if __name__ == "__main__":
    main()
