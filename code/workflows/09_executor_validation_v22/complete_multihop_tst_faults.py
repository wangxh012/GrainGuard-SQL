"""Complete unfiltered V2.2 TST faults over directed multi-hop FK paths.

The frozen contracts are read-only.  This module only fills fault-generation
gaps left by the legacy direct-join TST generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grain_test_sql.schema_registry import SchemaRegistry


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def key(row: dict[str, Any]) -> tuple[Any, Any, str, int]:
    return (
        row["question_id"],
        row["db_id"],
        row["contract_type"],
        int(row.get("contract_index", 0)),
    )


def directed_path_join_sql(
    source_table: str,
    group_table: str,
    registry: SchemaRegistry,
) -> tuple[str, str, str] | None:
    """Return FROM/JOIN SQL plus source/group aliases for child-to-parent paths."""
    path = registry.join_path(source_table, group_table)
    if not path or len(path) < 3:
        return None
    for child, parent in zip(path, path[1:]):
        if registry.fk_to_parent(child, parent) is None:
            return None

    aliases = {group_table: "g", source_table: "s"}
    intermediate = list(reversed(path[1:-1]))
    for index, table in enumerate(intermediate, start=1):
        aliases[table] = f"j{index}"

    clauses = [f"FROM {quote_ident(group_table)} AS g"]
    reversed_path = list(reversed(path))
    for parent, child in zip(reversed_path, reversed_path[1:]):
        fk = registry.fk_to_parent(child, parent)
        assert fk is not None
        child_alias = aliases[child]
        parent_alias = aliases[parent]
        predicates = [
            f"{child_alias}.{quote_ident(child_col)} = {parent_alias}.{quote_ident(parent_col)}"
            for child_col, parent_col in zip(fk.columns, fk.ref_columns)
        ]
        clauses.append(
            f"INNER JOIN {quote_ident(child)} AS {child_alias} ON " + " AND ".join(predicates)
        )
    return " ".join(clauses), aliases[source_table], aliases[group_table]


def build_multihop_tst_fault(
    wrapped_contract: dict[str, Any],
    registry: SchemaRegistry,
) -> str | None:
    if wrapped_contract.get("contract_type") != "TST":
        return None
    contract = wrapped_contract.get("contract") or {}
    measures = contract.get("measures") or []
    group = contract.get("output_group") or {}
    if len(measures) != 1 or not group.get("table") or not group.get("column"):
        return None
    measure = measures[0]
    if measure.get("ranking_metric_semantic") != "SUM_ONCE_PER_GRAIN":
        return None
    source_table = measure.get("source_table")
    source_column = measure.get("source_column")
    group_table = group["table"]
    group_column = group["column"]
    if not source_table or not source_column:
        return None
    joined = directed_path_join_sql(source_table, group_table, registry)
    if joined is None:
        return None
    from_clause, source_alias, group_alias = joined
    direction = str((contract.get("ranking") or {}).get("direction") or "DESC").upper()
    return (
        "/* grain_fault:synthetic v22_multihop_tst */ "
        f"SELECT {group_alias}.{quote_ident(group_column)} AS top_group, "
        f"MAX({source_alias}.{quote_ident(source_column)}) AS ranking_metric "
        f"{from_clause} "
        f"GROUP BY {group_alias}.{quote_ident(group_column)} "
        f"ORDER BY ranking_metric {direction} LIMIT 1"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill V2.2 multi-hop TST fault-generation gaps.")
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--contracts", required=True)
    parser.add_argument("--faults", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    eval_rows = read_jsonl(ROOT / args.eval_set)
    eval_by_key = {(row["question_id"], row["db_id"]): row for row in eval_rows}
    contract_rows = read_jsonl(ROOT / args.contracts)
    faults = read_jsonl(ROOT / args.faults)
    observed = {key(row) for row in faults}
    completions: list[dict[str, Any]] = []

    for row in contract_rows:
        contracts = row.get("auto_contracts") or []
        for contract_index, wrapped in enumerate(contracts):
            contract_key = (row["question_id"], row["db_id"], wrapped["contract_type"], contract_index)
            if contract_key in observed:
                continue
            eval_row = eval_by_key[(row["question_id"], row["db_id"])]
            registry = SchemaRegistry.from_yaml(ROOT / eval_row["schema_registry_path"])
            sql = build_multihop_tst_fault(wrapped, registry)
            if sql is None:
                continue
            completions.append(
                {
                    "question_id": row["question_id"],
                    "db_id": row["db_id"],
                    "candidate_sql": sql,
                    "candidate sql": sql,
                    "mutation": "unsafe_reverse_topk_metric",
                    "source": "v22_multihop_tst_synthetic",
                    "v22_fault_completion": "directed_multihop_tst",
                    "gold_sql": eval_row["gold_sql"],
                    "contract_type": "TST",
                    "contract_index": contract_index,
                    "generation_validation_status": None,
                }
            )
            observed.add(contract_key)

    combined = [*faults, *completions]
    write_jsonl(ROOT / args.out, combined)
    print(
        json.dumps(
            {
                "legacy_faults": len(faults),
                "multihop_tst_completions": len(completions),
                "completed_faults": len(combined),
                "completed_keys": [key(row) for row in completions],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
