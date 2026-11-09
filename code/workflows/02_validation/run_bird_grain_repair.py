from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.contract import MetricContract
from grain_test_sql.evidence_scope import EVIDENCE_SCOPE_POLICIES, SUPPORTED_POLICY
from grain_test_sql.gold_semantic_audit import (
    apply_gold_semantic_audit,
    audit_gold_sql_semantics,
)
from grain_test_sql.pipeline import run_pipeline
from grain_test_sql.prompts import build_repair_prompt
from grain_test_sql.schema_registry import SchemaRegistry
from grain_test_sql.bird_utils import (
    calibrate_registry_filter_uniqueness_from_sql,
    calibrate_registry_uniqueness_from_sqlite,
    read_jsonl_lenient,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", default="artifacts/datasets/bird/eval/bird_eval_set.jsonl")
    parser.add_argument("--predictions", default="", help="Optional JSONL with question_id, db_id, candidate sql")
    parser.add_argument("--auto-contracts", default="", help="Optional JSONL from generate_bird_auto_contracts.py")
    parser.add_argument(
        "--auto-contract-row-set",
        action="store_true",
        help="Validate only rows present in --auto-contracts, preserving the auto-contract file order before --limit",
    )
    parser.add_argument("--contract-source", choices=["silver", "auto"], default="silver")
    parser.add_argument("--limit", type=int, default=0, help="Only validate the first N eval rows after loading the eval set")
    parser.add_argument(
        "--question-ids",
        default="",
        help="Comma-separated question_id list to validate only selected rows before applying --limit",
    )
    parser.add_argument("--resume", action="store_true", help="Append missing validation rows and keep existing output rows")
    parser.add_argument("--only-with-contract", action="store_true", help="Skip rows without contracts before applying --limit")
    parser.add_argument(
        "--only-predicted-contracts",
        action="store_true",
        help="When predictions are contract-specific, validate only contract keys present in --predictions.",
    )
    parser.add_argument(
        "--contract-confidence",
        choices=["all", "high"],
        default="all",
        help="For silver contracts, optionally keep only high-confidence contracts for Gold dry-run false-positive checks.",
    )
    parser.add_argument("--out", default="artifacts/runs/bird/glm-5.1/03_model_prediction_validation/grain_repair_results.jsonl")
    parser.add_argument("--use-gold-as-candidate", action="store_true")
    parser.add_argument("--bird-dir", default="artifacts/datasets/bird/raw/dev_20240627", help="BIRD dev directory for real SQLite result comparison")
    parser.add_argument(
        "--skip-real-result-compare",
        action="store_true",
        help="Do not execute gold and candidate SQL against the real SQLite database",
    )
    parser.add_argument(
        "--real-result-row-limit",
        type=int,
        default=200,
        help="Rows to keep in each recorded real result preview; use 0 to keep all rows",
    )
    parser.add_argument(
        "--real-result-timeout-sec",
        type=float,
        default=20.0,
        help="Per-query timeout for real SQLite result comparison",
    )
    parser.add_argument("--generate-repairs", action="store_true", help="Call the configured LLM to repair VIOLATION rows")
    parser.add_argument(
        "--evidence-scope-policy",
        choices=EVIDENCE_SCOPE_POLICIES,
        default=SUPPORTED_POLICY,
        help="Evidence admission policy; controlled executor replay explicitly uses unrestricted.",
    )
    parser.add_argument(
        "--guard-static-pass-violations",
        action="store_true",
        help="Downgrade executable witness violations when static grain rules find no high-confidence anti-pattern",
    )
    parser.add_argument(
        "--enable-static-grain-alerts",
        action="store_true",
        help="Promote high-confidence static grain anti-patterns to VIOLATION even when executable witness is unavailable or schema-gated.",
    )
    parser.add_argument(
        "--calibrate-registry-uniqueness-from-db",
        action="store_true",
        help="Use the real SQLite DB to mark observed non-unique columns before schema-pass decisions.",
    )
    parser.add_argument(
        "--calibrate-sql-filter-uniqueness-from-db",
        action="store_true",
        help=(
            "Use the real SQLite DB to add SQL-specific filtered uniqueness facts. "
            "This is intended for candidate prediction validation; GOLD dry-runs usually "
            "need the stricter semantic-risk behavior."
        ),
    )
    parser.add_argument(
        "--sqlite-db-root",
        default="",
        help="Optional root containing db_id/db_id.sqlite; used for registry uniqueness calibration.",
    )
    parser.add_argument(
        "--gold-question-sql-audit",
        action="store_true",
        help=(
            "For Spider GOLD dry-runs only, apply question-aware SQL audit rules that can suppress "
            "contract-boundary false positives and flag missed intrinsic GOLD SQL errors."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    eval_rows = read_jsonl(Path(args.eval_set))
    auto_contract_rows = read_jsonl(Path(args.auto_contracts)) if args.auto_contracts else []
    if args.auto_contract_row_set:
        eval_rows = filter_eval_rows_by_contract_order(eval_rows, auto_contract_rows)
    question_ids = parse_question_ids(args.question_ids)
    if question_ids:
        eval_rows = [row for row in eval_rows if row.get("question_id") in question_ids]
    auto_contracts = contracts_by_key(auto_contract_rows, "auto_contracts") if auto_contract_rows else {}
    if args.only_with_contract:
        eval_rows = [
            row
            for row in eval_rows
            if row.get("silver_contracts")
            or row.get("auto_contracts")
            or auto_contracts.get((row.get("question_id"), row.get("db_id")))
        ]
    if args.limit:
        eval_rows = eval_rows[: args.limit]
    gold_question_sql_audit_enabled = (
        args.gold_question_sql_audit and "spider" in str(args.eval_set).lower()
    )
    predictions, predictions_are_contract_specific = (
        load_predictions(Path(args.predictions)) if args.predictions else ({}, False)
    )
    existing_rows: list[dict[str, Any]] = []
    done_keys: set[tuple[Any, ...]] = set()
    if args.resume and Path(args.out).exists():
        existing_rows = read_jsonl(Path(args.out))
        for row in existing_rows:
            if row.get("question_id") is None or row.get("db_id") is None or row.get("contract_type") is None:
                continue
            key: tuple[Any, ...] = (row["question_id"], row["db_id"], row["contract_type"])
            if "contract_index" in row:
                key = (*key, row["contract_index"])
            done_keys.add(key)
    out_rows: list[dict[str, Any]] = []
    llm = build_llm_client(args) if args.generate_repairs else None
    stats = {
        "rows": len(eval_rows),
        "with_contract": 0,
        "checked": 0,
        "violations": 0,
        "passes": 0,
        "executable_passes": 0,
        "schema_passes": 0,
        "static_passes": 0,
        "vacuous_passes": 0,
        "total_passes": 0,
        "invalid_witnesses": 0,
        "unsupported": 0,
        "missing_candidate": 0,
        "skipped_low_confidence_contracts": 0,
        "real_result_compared": 0,
        "real_result_matches": 0,
        "real_result_ordered_matches": 0,
        "real_result_bag_matches": 0,
        "real_result_mismatches": 0,
        "real_result_errors": 0,
        "real_result_failures": 0,
        "target_bug_checked": 0,
        "target_bug_violations": 0,
        "target_bug_passes": 0,
        "gold_dry_run_checked": 0,
        "gold_dry_run_violations": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "repair_real_result_compared": 0,
        "repair_real_result_matches": 0,
        "registry_uniqueness_calibrated": 0,
        "registry_uniqueness_calibration_missing_db": 0,
        "registry_filter_uniqueness_calibrated": 0,
    }

    registries: dict[tuple[str, str], SchemaRegistry] = {}
    for row in eval_rows:
        key = (row["question_id"], row["db_id"])
        if args.contract_source == "auto":
            contracts = auto_contracts.get(key) or row.get("auto_contracts") or []
        else:
            contracts = row.get("silver_contracts") or []
        contracts = [dict(contract, _contract_index=index) for index, contract in enumerate(contracts)]
        if args.contract_source == "silver" and args.contract_confidence == "high":
            before = len(contracts)
            contracts = [contract for contract in contracts if is_high_confidence_silver(contract)]
            stats["skipped_low_confidence_contracts"] += before - len(contracts)
        if args.only_predicted_contracts and predictions_are_contract_specific:
            contracts = [
                contract
                for contract in contracts
                if prediction_exists_for_contract(
                    predictions,
                    row["question_id"],
                    row["db_id"],
                    contract,
                )
            ]
        if not contracts:
            continue
        stats["with_contract"] += 1
        print(
            f"question_id={row['question_id']} db_id={row['db_id']} contracts={len(contracts)}",
            flush=True,
        )
        registry_path = row["schema_registry_path"]
        sqlite_path = resolve_sqlite_path(args, row) if args.calibrate_registry_uniqueness_from_db else None
        registry_cache_key = (registry_path, str(sqlite_path or ""))
        if registry_cache_key not in registries:
            registry = SchemaRegistry.from_yaml(registry_path)
            if args.calibrate_registry_uniqueness_from_db:
                if sqlite_path and sqlite_path.exists():
                    calibrate_registry_uniqueness_from_sqlite(
                        registry,
                        sqlite_path,
                        profile_all_single_columns=True,
                    )
                    stats["registry_uniqueness_calibrated"] += 1
                else:
                    stats["registry_uniqueness_calibration_missing_db"] += 1
            registries[registry_cache_key] = registry
        registry = registries[registry_cache_key]
        for contract_index, wrapped_contract in enumerate(contracts):
            done_key: tuple[Any, ...] = (row["question_id"], row["db_id"], wrapped_contract["contract_type"])
            if done_key in done_keys or (*done_key, wrapped_contract.get("_contract_index", contract_index)) in done_keys:
                print(
                    f"  skip existing contract_type={wrapped_contract['contract_type']} "
                    f"contract_index={wrapped_contract.get('_contract_index', contract_index)}",
                    flush=True,
                )
                continue
            contract_index_key = (
                row["question_id"],
                row["db_id"],
                wrapped_contract["contract_type"],
                wrapped_contract.get("_contract_index", contract_index),
            )
            contract_key = (row["question_id"], row["db_id"], wrapped_contract["contract_type"])
            prediction_row = predictions.get(contract_index_key) or predictions.get(contract_key)
            if not predictions_are_contract_specific:
                prediction_row = prediction_row or predictions.get(key)
            candidate_sql = prediction_candidate_sql(prediction_row)
            candidate_source = prediction_source(prediction_row)
            mutation = prediction_mutation(prediction_row)
            if not candidate_sql and args.use_gold_as_candidate:
                candidate_sql = row["gold_sql"]
                candidate_source = "gold_dry_run"
                mutation = None
            if not candidate_sql:
                stats["missing_candidate"] += 1
                print(
                    f"  missing_candidate contract_type={wrapped_contract['contract_type']} "
                    f"contract_index={wrapped_contract.get('_contract_index', contract_index)}",
                    flush=True,
                )
                out_rows.append(
                    {
                        "question_id": row["question_id"],
                        "db_id": row["db_id"],
                        "question": row.get("question"),
                        "evidence": row.get("evidence"),
                        "contract_type": wrapped_contract["contract_type"],
                        "contract_index": wrapped_contract.get("_contract_index", contract_index),
                        "contract_source": args.contract_source,
                        "contract": wrapped_contract.get("contract"),
                        "candidate_source": candidate_source,
                        "mutation": mutation,
                        "candidate sql": "",
                        "gold_sql": row["gold_sql"],
                        "real_result_comparison": None,
                        "grain_result": {
                            "status": "MISSING_CANDIDATE",
                            "results": [],
                            "gate_errors": ["missing candidate SQL"],
                            "repair_hints": [],
                            "pass_class": None,
                        },
                        "repair_prompt": None,
                        "repaired_sql": None,
                        "repaired_grain_result": None,
                        "repaired_real_result_comparison": None,
                    }
                )
                continue
            contract = MetricContract.from_dict(wrapped_contract["contract"])
            pipeline_registry = registry
            if args.calibrate_sql_filter_uniqueness_from_db and sqlite_path and sqlite_path.exists():
                # SQL-filter uniqueness is query-local. Keep it off the cached
                # registry so one question's filters cannot change another
                # question's validation behavior in the same database.
                pipeline_registry = copy.deepcopy(registry)
                calibrate_registry_filter_uniqueness_from_sql(pipeline_registry, sqlite_path, candidate_sql)
                stats["registry_filter_uniqueness_calibrated"] += 1
            pipeline_error = None
            try:
                result = run_pipeline(
                    candidate_sql,
                    pipeline_registry,
                    contract,
                    guard_static_pass_violations=args.guard_static_pass_violations,
                    enable_static_alerts=args.enable_static_grain_alerts,
                    evidence_scope_policy=args.evidence_scope_policy,
                ).to_dict()
            except Exception as exc:
                pipeline_error = exc
                result = pipeline_error_result(exc)
            if gold_question_sql_audit_enabled and candidate_source == "gold_dry_run":
                audit_decision = audit_gold_sql_semantics(
                    row.get("question") or "",
                    candidate_sql,
                    current_status=result.get("status"),
                )
                if audit_decision is not None:
                    result = apply_gold_semantic_audit(result, audit_decision)
            real_result_comparison = None
            if not args.skip_real_result_compare:
                real_result_comparison = compare_real_results(
                    Path(args.bird_dir),
                    row["db_id"],
                    row["gold_sql"],
                    candidate_sql,
                    args.real_result_row_limit,
                    args.real_result_timeout_sec,
                )
                stats["real_result_compared"] += 1
                has_real_result_error = (
                    real_result_comparison["gold"].get("error")
                    or real_result_comparison["candidate"].get("error")
                )
                if real_result_comparison["value_ordered_match"]:
                    stats["real_result_ordered_matches"] += 1
                if has_real_result_error:
                    stats["real_result_errors"] += 1
                    stats["real_result_failures"] += 1
                elif real_result_comparison["value_bag_match"]:
                    stats["real_result_matches"] += 1
                    stats["real_result_bag_matches"] += 1
                else:
                    stats["real_result_mismatches"] += 1
                    stats["real_result_failures"] += 1
            repair_prompt = (
                None
                if pipeline_error is not None
                else build_repair_prompt(row, candidate_sql, result, contract, registry)
            )
            repaired_sql = None
            repaired_grain_result = None
            repaired_real_result_comparison = None
            if repair_prompt and llm:
                repaired_sql = extract_sql(
                    llm.chat(
                        [{"role": "user", "content": repair_prompt}],
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        enable_thinking=False,
                    )
                )
                repaired_grain_result = run_pipeline(
                    repaired_sql,
                    registry,
                    contract,
                    guard_static_pass_violations=args.guard_static_pass_violations,
                    enable_static_alerts=args.enable_static_grain_alerts,
                    evidence_scope_policy=args.evidence_scope_policy,
                ).to_dict()
                if not args.skip_real_result_compare:
                    repaired_real_result_comparison = compare_real_results(
                        Path(args.bird_dir),
                        row["db_id"],
                        row["gold_sql"],
                        repaired_sql,
                        args.real_result_row_limit,
                        args.real_result_timeout_sec,
                    )
                    stats["repair_real_result_compared"] += 1
                    if repaired_real_result_comparison["value_bag_match"]:
                        stats["repair_real_result_matches"] += 1
            status = result["status"]
            stats["checked"] += 1
            if candidate_source in {
                "gold_sql_fault_injection",
                "contract_synthetic_negative",
                "contract_synthetic_forced",
            }:
                stats["target_bug_checked"] += 1
                if status == "VIOLATION":
                    stats["target_bug_violations"] += 1
                elif status == "PASS":
                    stats["target_bug_passes"] += 1
            if candidate_source == "gold_dry_run":
                stats["gold_dry_run_checked"] += 1
                if status == "VIOLATION":
                    stats["gold_dry_run_violations"] += 1
            if repaired_grain_result:
                stats["repair_attempts"] += 1
                repair_grain_passed = repaired_grain_result.get("status") == "PASS"
                repair_result_matched = (
                    args.skip_real_result_compare
                    or (
                        repaired_real_result_comparison is not None
                        and repaired_real_result_comparison.get("value_bag_match")
                    )
                )
                if status == "VIOLATION" and repair_grain_passed and repair_result_matched:
                    stats["repair_successes"] += 1
            if status == "VIOLATION":
                stats["violations"] += 1
            elif status == "PASS":
                stats["total_passes"] += 1
                pass_class = result.get("pass_class")
                if pass_class == "EXECUTABLE_PASS":
                    stats["passes"] += 1
                    stats["executable_passes"] += 1
                elif pass_class == "SCHEMA_PASS":
                    stats["passes"] += 1
                    stats["schema_passes"] += 1
                elif pass_class == "VACUOUS_PASS":
                    stats["vacuous_passes"] += 1
                else:
                    stats["static_passes"] += 1
            elif status == "INVALID_WITNESS":
                stats["invalid_witnesses"] += 1
            else:
                stats["unsupported"] += 1
            print(
                f"  result contract_type={wrapped_contract['contract_type']} "
                f"contract_index={wrapped_contract.get('_contract_index', contract_index)} "
                f"status={status}",
                flush=True,
            )
            out_rows.append(
                {
                    "question_id": row["question_id"],
                    "db_id": row["db_id"],
                    "question": row.get("question"),
                    "evidence": row.get("evidence"),
                    "contract_type": wrapped_contract["contract_type"],
                    "contract_index": wrapped_contract.get("_contract_index", contract_index),
                    "contract_source": args.contract_source,
                    "contract": wrapped_contract.get("contract"),
                    "candidate_source": candidate_source,
                    "mutation": mutation,
                    "candidate sql": candidate_sql,
                    "gold_sql": row["gold_sql"],
                    "real_result_comparison": real_result_comparison,
                    "grain_result": result,
                    "evidence_scope_policy": args.evidence_scope_policy,
                    "repair_prompt": repair_prompt,
                    "repaired_sql": repaired_sql,
                    "repaired_grain_result": repaired_grain_result,
                    "repaired_real_result_comparison": repaired_real_result_comparison,
                }
            )

    output_rows = existing_rows + out_rows
    stats.update(derived_experiment_metrics(stats))
    write_jsonl(Path(args.out), output_rows)
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_lenient(path)


def pipeline_error_result(exc: Exception) -> dict[str, Any]:
    return {
        "status": "UNSUPPORTED",
        "results": [
            {
                "status": "UNSUPPORTED",
                "pass_class": None,
                "test_type": None,
                "measure": None,
                "diagnosis": f"pipeline error: {type(exc).__name__}: {exc}",
            }
        ],
        "gate_errors": [],
        "repair_hints": [],
        "pass_class": None,
    }


def load_predictions(path: Path) -> tuple[dict[tuple[Any, ...], dict[str, Any]], bool]:
    rows = read_jsonl(path)
    predictions: dict[tuple[Any, ...], dict[str, Any]] = {}
    has_contract_specific = any(row.get("contract_type") for row in rows)
    for row in rows:
        key = (row["question_id"], row["db_id"])
        candidate_sql = row.get("candidate sql") or row.get("candidate_sql")
        if not candidate_sql:
            continue
        if row.get("contract_type"):
            if "contract_index" in row:
                predictions[(*key, row["contract_type"], row["contract_index"])] = row
            predictions[(*key, row["contract_type"])] = row
        else:
            predictions[key] = row
    return predictions, has_contract_specific


def prediction_candidate_sql(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    return row.get("candidate sql") or row.get("candidate_sql")


def prediction_source(row: dict[str, Any] | None) -> str:
    if not row:
        return "prediction"
    return row.get("source") or "prediction"


def prediction_mutation(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    return row.get("mutation")


def prediction_exists_for_contract(
    predictions: dict[tuple[Any, ...], dict[str, Any]],
    question_id: Any,
    db_id: Any,
    contract: dict[str, Any],
) -> bool:
    contract_type = contract.get("contract_type")
    contract_index = contract.get("_contract_index", 0)
    return (
        (question_id, db_id, contract_type, contract_index) in predictions
        or (question_id, db_id, contract_type) in predictions
    )


def derived_experiment_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    target_checked = stats["target_bug_checked"]
    gold_checked = stats["gold_dry_run_checked"]
    repair_attempts = stats["repair_attempts"]
    baseline_unsafe_commits = target_checked
    after_gate_unsafe_commits = target_checked - stats["target_bug_violations"]
    return {
        "target_bug_recall": safe_rate(stats["target_bug_violations"], target_checked),
        "false_positive_rate": safe_rate(stats["gold_dry_run_violations"], gold_checked),
        "repair_success_rate": safe_rate(stats["repair_successes"], repair_attempts),
        "baseline_unsafe_commits": baseline_unsafe_commits,
        "after_gate_unsafe_commits": after_gate_unsafe_commits,
        "unsafe_commit_reduction": safe_rate(
            baseline_unsafe_commits - after_gate_unsafe_commits,
            baseline_unsafe_commits,
        ),
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def parse_question_ids(value: str) -> set[int]:
    ids: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.add(int(part))
    return ids


def resolve_sqlite_path(args: argparse.Namespace, row: dict[str, Any]) -> Path | None:
    db_id = row.get("db_id")
    if not db_id:
        return None
    explicit = row.get("sqlite_path")
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path

    roots: list[Path] = []
    if args.sqlite_db_root:
        roots.append(Path(args.sqlite_db_root))
    if args.bird_dir:
        roots.append(Path(args.bird_dir) / "dev_databases")
    roots.extend(
        [
            ROOT / "artifacts" / "datasets" / "bird" / "raw" / "dev_20240627" / "dev_databases",
            ROOT / "artifacts" / "datasets" / "spider" / "spider_data" / "spider_data" / "database",
        ]
    )
    seen: set[Path] = set()
    for root in roots:
        root = root.expanduser()
        if root in seen:
            continue
        seen.add(root)
        path = root / db_id / f"{db_id}.sqlite"
        if path.exists():
            return path
    return None


def load_contracts(path: Path, field: str) -> dict[tuple[int, str], list[dict[str, Any]]]:
    return contracts_by_key(read_jsonl(path), field)


def contracts_by_key(rows: list[dict[str, Any]], field: str) -> dict[tuple[int, str], list[dict[str, Any]]]:
    contracts: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        contracts[(row["question_id"], row["db_id"])] = row.get(field) or []
    return contracts


def filter_eval_rows_by_contract_order(
    eval_rows: list[dict[str, Any]],
    contract_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eval_by_key = {
        (row["question_id"], row["db_id"]): row
        for row in eval_rows
    }
    return [
        eval_by_key[(row["question_id"], row["db_id"])]
        for row in contract_rows
        if (row.get("question_id"), row.get("db_id")) in eval_by_key
    ]


def is_high_confidence_silver(contract: dict[str, Any]) -> bool:
    tier = contract.get("silver_tier")
    if tier:
        return tier == "high"
    confidence = contract.get("confidence", "")
    return bool(confidence) and not confidence.startswith("weak_")


def build_llm_client(args: argparse.Namespace) -> Any:
    from grain_test_sql.llm_client import LLMConfig, OpenAICompatibleClient

    return OpenAICompatibleClient(
        LLMConfig.from_values(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            timeout=getattr(args, "llm_timeout", None),
        )
    )


def compare_real_results(
    bird_dir: Path,
    db_id: str,
    gold_sql: str,
    candidate_sql: str,
    row_limit: int,
    timeout_sec: float,
) -> dict[str, Any]:
    db_path = bird_dir / "dev_databases" / db_id / f"{db_id}.sqlite"
    gold = execute_real_sql(db_path, gold_sql, row_limit, timeout_sec)
    candidate = execute_real_sql(db_path, candidate_sql, row_limit, timeout_sec)
    ordered_match = (
        not gold.get("error")
        and not candidate.get("error")
        and gold.get("full_result_sha256") == candidate.get("full_result_sha256")
    )
    bag_match = (
        not gold.get("error")
        and not candidate.get("error")
        and gold.get("bag_result_sha256") == candidate.get("bag_result_sha256")
    )
    value_ordered_match = (
        not gold.get("error")
        and not candidate.get("error")
        and gold.get("value_result_sha256") == candidate.get("value_result_sha256")
    )
    value_bag_match = (
        not gold.get("error")
        and not candidate.get("error")
        and gold.get("value_bag_result_sha256") == candidate.get("value_bag_result_sha256")
    )
    return {
        "db_path": str(db_path),
        "exact_match": value_ordered_match,
        "ordered_match": ordered_match,
        "bag_match": bag_match,
        "value_ordered_match": value_ordered_match,
        "value_bag_match": value_bag_match,
        "gold": gold,
        "candidate": candidate,
    }


def execute_real_sql(db_path: Path, sql: str, row_limit: int, timeout_sec: float) -> dict[str, Any]:
    if not db_path.exists():
        return {
            "error": f"database not found: {db_path}",
            "columns": [],
            "row_count": 0,
            "rows": [],
            "truncated": False,
            "full_result_sha256": None,
        }
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    deadline = time.monotonic() + timeout_sec

    def check_timeout() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(check_timeout, 10_000)
    try:
        cursor = conn.execute(sql)
        columns = [description[0] for description in cursor.description or []]
        fetched_rows = cursor.fetchall()
        rows = [dict(row) for row in fetched_rows]
        value_rows = [list(row) for row in fetched_rows]
    except Exception as exc:
        return {
            "error": str(exc),
            "columns": [],
            "row_count": 0,
            "rows": [],
            "truncated": False,
            "full_result_sha256": None,
        }
    finally:
        conn.close()

    full_payload = {"columns": columns, "rows": rows}
    full_json = json.dumps(full_payload, ensure_ascii=False, sort_keys=True, default=str)
    bag_rows = sorted(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) for row in rows)
    )
    bag_json = json.dumps({"columns": columns, "rows": bag_rows}, ensure_ascii=False, sort_keys=True)
    value_json = json.dumps(value_rows, ensure_ascii=False, default=str)
    value_bag_json = json.dumps(
        sorted(json.dumps(row, ensure_ascii=False, default=str) for row in value_rows),
        ensure_ascii=False,
    )
    keep_rows = rows if row_limit == 0 else rows[:row_limit]
    keep_value_rows = value_rows if row_limit == 0 else value_rows[:row_limit]
    return {
        "error": None,
        "columns": columns,
        "row_count": len(rows),
        "rows": keep_rows,
        "value_rows": keep_value_rows,
        "truncated": row_limit != 0 and len(rows) > row_limit,
        "full_result_sha256": hashlib.sha256(full_json.encode("utf-8")).hexdigest(),
        "bag_result_sha256": hashlib.sha256(bag_json.encode("utf-8")).hexdigest(),
        "value_result_sha256": hashlib.sha256(value_json.encode("utf-8")).hexdigest(),
        "value_bag_result_sha256": hashlib.sha256(value_bag_json.encode("utf-8")).hexdigest(),
    }


def extract_sql(text: str) -> str:
    import re

    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    text = text.strip()
    text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


if __name__ == "__main__":
    main()
