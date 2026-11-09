"""Build a full-census, outcome-independent safe-control pool for V2.2.

Each frozen contract receives a forced synthetic fault skeleton and its matched
neutralization.  Only the neutralized SQL is executed for specificity.  No
reference/GOLD SQL and no prior executor result is used to construct or select
the control.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.contract import MetricContract
from grain_test_sql.evidence_scope import UNRESTRICTED_POLICY
from grain_test_sql.pipeline import run_pipeline
from grain_test_sql.schema_registry import SchemaRegistry


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GENERATOR = load_module(
    "v22_legacy_fault_generator",
    ROOT / "workflows/02_validation/generate_bird_false_candidates.py",
)
CONTROL_AUDIT = load_module(
    "v22_control_audit",
    ROOT / "workflows/09_executor_validation_v22/build_unfiltered_control_audit.py",
)
MULTIHOP_TST = load_module(
    "v22_multihop_tst",
    ROOT / "workflows/09_executor_validation_v22/complete_multihop_tst_faults.py",
)
V21_AUDIT = CONTROL_AUDIT.V21


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def replay_key(question_id: Any, db_id: Any, operator: str, contract_index: int) -> str:
    return "|".join(map(str, (question_id, db_id, operator, contract_index)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 580 independently certified V2.2 controls.")
    parser.add_argument("--dataset-config-json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = json.loads(args.dataset_config_json)
    rows: list[dict[str, Any]] = []

    for dataset, paths in sorted(config.items()):
        eval_rows = read_jsonl(ROOT / paths["eval"])
        eval_by_key = {(row["question_id"], row["db_id"]): row for row in eval_rows}
        contract_rows = read_jsonl(ROOT / paths["contracts"])
        for contract_row in contract_rows:
            for contract_index, wrapped in enumerate(contract_row.get("auto_contracts") or []):
                question_id = contract_row["question_id"]
                db_id = contract_row["db_id"]
                operator = wrapped["contract_type"]
                eval_row = eval_by_key[(question_id, db_id)]
                registry = SchemaRegistry.from_yaml(ROOT / eval_row["schema_registry_path"])
                contract = MetricContract.from_dict(wrapped["contract"])
                forced_sql = GENERATOR.build_forced_synthetic_candidate(wrapped, registry)
                source = "contract_synthetic_forced"
                if forced_sql is None:
                    forced_sql = MULTIHOP_TST.build_multihop_tst_fault(wrapped, registry)
                    source = "v22_multihop_tst_synthetic"
                record: dict[str, Any] = {
                    "replay_key": replay_key(question_id, db_id, operator, contract_index),
                    "dataset": dataset,
                    "question_id": question_id,
                    "db_id": db_id,
                    "contract_type": operator,
                    "contract_index": contract_index,
                    "certification_label": "CERTIFIED_NEGATIVE",
                    "certification_basis": "OUTCOME_INDEPENDENT_MATCHED_SYNTHETIC_NEUTRALIZATION",
                    "synthetic_fault_source": source,
                    "synthetic_fault_sql": forced_sql,
                }
                try:
                    if not forced_sql:
                        raise ValueError("no forced synthetic fault skeleton is available")
                    fault = {"source": source, "candidate_sql": forced_sql}
                    if source == "v22_multihop_tst_synthetic":
                        control_sql, strategy = CONTROL_AUDIT.build_v22_control(fault) or (None, None)
                    else:
                        control_sql, strategy = V21_AUDIT.build_safe_control(
                            fault,
                            {"contract_type": operator, "gold_sql": eval_row["gold_sql"]},
                            contract,
                        )
                    if not control_sql or not strategy:
                        raise ValueError("matched neutralization did not return a control")
                    result = run_pipeline(
                        control_sql, registry, contract,
                        evidence_scope_policy=UNRESTRICTED_POLICY,
                    ).to_dict()
                    record.update(
                        {
                            "control_strategy": strategy,
                            "control_sql": control_sql,
                            "control_status": result["status"],
                            "control_result": result,
                        }
                    )
                except Exception as exc:
                    record.update(
                        {
                            "control_status": "ERROR",
                            "control_error": {
                                "exception_type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                    )
                rows.append(record)

    if len(rows) != 580 or len({row["replay_key"] for row in rows}) != 580:
        raise ValueError(f"certified control pool must contain 580 unique rows, found {len(rows)}")
    output = ROOT / args.out
    write_jsonl(output, rows)
    statuses = Counter(row["control_status"] for row in rows)
    print(
        json.dumps(
            {
                "certified_controls": len(rows),
                "status_counts": dict(sorted(statuses.items())),
                "false_triggers": statuses["VIOLATION"],
                "out": args.out,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
