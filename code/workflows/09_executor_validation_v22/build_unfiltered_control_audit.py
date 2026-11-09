"""Build V2.2 controls and witness audits without excluding failed replay rows.

The legacy controlled-validity builder is reused only for its pure pair/control
construction routines.  Unlike its report path, this driver records every
construction or audit failure as data, so no denominator is silently reduced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.contract import MetricContract
from grain_test_sql.evidence_scope import UNRESTRICTED_POLICY
from grain_test_sql.pipeline import run_pipeline
from grain_test_sql.schema_registry import SchemaRegistry


LEGACY = ROOT / "workflows/07_controlled_validity/build_controlled_validity_audit.py"
SPEC = importlib.util.spec_from_file_location("v21_controlled_validity", LEGACY)
assert SPEC and SPEC.loader
V21 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V21)


def row_key(row: dict[str, Any]) -> tuple[Any, Any, str, int]:
    return (
        row["question_id"],
        row["db_id"],
        row["contract_type"],
        int(row.get("contract_index", 0)),
    )


def serial_key(key: tuple[Any, Any, str, int]) -> str:
    return "|".join(map(str, key))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def error_detail(exc: Exception) -> dict[str, str]:
    return {"exception_type": type(exc).__name__, "message": str(exc)}


def build_v22_control(fault: dict[str, Any]) -> tuple[str, str] | None:
    """Neutralize a V2.2-only TST MAX-for-SUM fault without changing its path."""
    if fault.get("source") != "v22_multihop_tst_synthetic":
        return None
    sql = fault.get("candidate_sql") or fault.get("candidate sql")
    tree = sqlglot.parse_one(sql, read="sqlite")
    maximum = next(tree.find_all(exp.Max), None)
    if maximum is None:
        raise ValueError("V2.2 multi-hop TST fault has no MAX ranking metric")
    maximum.replace(exp.Sum(this=maximum.this.copy()))
    return tree.sql(dialect="sqlite"), "restore_contract_sum_on_same_multihop_path"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unfiltered V2.2 controls and witness audit records.")
    parser.add_argument("--dataset-config-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(args.dataset_config_json)
    if not isinstance(config, dict) or not config:
        raise ValueError("--dataset-config-json must be a non-empty dataset mapping")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    controls: list[dict[str, Any]] = []
    witnesses: list[dict[str, Any]] = []

    for dataset, paths in sorted(config.items()):
        if not {"eval", "faults", "results"} <= set(paths):
            raise ValueError(f"{dataset} must contain eval, faults, and results paths")
        eval_rows = V21.index_eval_rows(read_jsonl(ROOT / paths["eval"]))
        faults = read_jsonl(ROOT / paths["faults"])
        results = read_jsonl(ROOT / paths["results"])
        faults_by_key = {row_key(row): row for row in faults}
        if len(faults_by_key) != len(faults):
            raise ValueError(f"{dataset}: duplicate fault keys")
        result_keys = {row_key(row) for row in results}
        if result_keys != set(faults_by_key):
            missing = sorted(set(faults_by_key) - result_keys)
            extra = sorted(result_keys - set(faults_by_key))
            raise ValueError(f"{dataset}: fault/result key mismatch; missing={missing[:3]} extra={extra[:3]}")

        registries: dict[str, SchemaRegistry] = {}
        for result in results:
            key = row_key(result)
            fault = faults_by_key[key]
            contract = MetricContract.from_dict(result["contract"])
            eval_row = eval_rows[(result["question_id"], result["db_id"])]
            registry_path = str(ROOT / eval_row["schema_registry_path"])
            registry = registries.setdefault(registry_path, SchemaRegistry.from_yaml(registry_path))
            control = {
                "replay_key": serial_key(key),
                "dataset": dataset,
                "question_id": result["question_id"],
                "db_id": result["db_id"],
                "contract_type": result["contract_type"],
                "contract_index": key[-1],
            }
            try:
                v22_control = build_v22_control(fault)
                if v22_control is None:
                    safe_sql, strategy = V21.build_safe_control(fault, result, contract)
                else:
                    safe_sql, strategy = v22_control
                safe_result = run_pipeline(
                    safe_sql, registry, contract,
                    evidence_scope_policy=UNRESTRICTED_POLICY,
                ).to_dict()
                control.update(
                    {
                        "control_status": safe_result["status"],
                        "control_strategy": strategy,
                        "safe_sql": safe_sql,
                        "control_result": safe_result,
                    }
                )
            except Exception as exc:  # record incompatibility rather than pruning it
                control.update({"control_status": "ERROR", "control_error": error_detail(exc)})
            controls.append(control)

            fault_status = str((result.get("grain_result") or {}).get("status") or "ERROR")
            witness = {
                "replay_key": serial_key(key),
                "dataset": dataset,
                "question_id": result["question_id"],
                "db_id": result["db_id"],
                "contract_type": result["contract_type"],
                "contract_index": key[-1],
                "fault_status": fault_status,
                "triggered": fault_status == "VIOLATION",
            }
            if fault_status != "VIOLATION":
                witness["reconstruction_status"] = "NOT_TRIGGERED"
            else:
                try:
                    reconstructed = V21.reconstruct_witnesses(
                        dataset, result, registry_path, registry, contract
                    )
                    if len(reconstructed) != 1:
                        raise ValueError(f"expected one retained witness, received {len(reconstructed)}")
                    precheck = V21.independent_precheck(reconstructed[0])
                    witness.update(
                        {
                            "reconstruction_status": "RECONSTRUCTED",
                            "audit_id": precheck["audit_id"],
                            "code_audit_valid": bool(precheck["code_audit_valid"]),
                            "code_audit": precheck,
                        }
                    )
                except Exception as exc:  # a failed witness remains in the validity denominator
                    witness.update(
                        {
                            "reconstruction_status": "RECONSTRUCTION_ERROR",
                            "code_audit_valid": False,
                            "witness_error": error_detail(exc),
                        }
                    )
            witnesses.append(witness)

    write_jsonl(output_dir / "controls.unfiltered.jsonl", controls)
    write_jsonl(output_dir / "witness_audit.unfiltered.jsonl", witnesses)
    print(json.dumps({"controls": len(controls), "witness_records": len(witnesses)}, indent=2))


if __name__ == "__main__":
    main()
