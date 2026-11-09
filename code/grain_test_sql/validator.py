from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Any

import sqlglot
from sqlglot import exp

from .contract_checker import contract_value
from .executor import execute_sql
from .pair_builder import TestPair
from .schema_registry import SchemaRegistry


@dataclass
class ValidationResult:
    status: str
    test_type: str
    measure: str
    pass_class: str | None
    contract_value_d0: Any
    contract_value_d1: Any
    sql_output_d0: Any
    sql_output_d1: Any
    diagnosis: str
    contract_preserved: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "test_type": self.test_type,
            "measure": self.measure,
            "pass_class": self.pass_class,
            "contract_value_D0": self.contract_value_d0,
            "contract_value_D1": self.contract_value_d1,
            "sql_output_D0": self.sql_output_d0,
            "sql_output_D1": self.sql_output_d1,
            "diagnosis": self.diagnosis,
        }
        if self.contract_preserved is not None:
            result["contract_preserved"] = self.contract_preserved
        return result


def validate_pair(
    sql: str,
    pair: TestPair,
    registry: SchemaRegistry,
    *,
    require_contract_preservation: bool = True,
    contract_preserved: bool | None = None,
    output_comparator: str = "exact",
) -> ValidationResult:
    out0 = _execute_sql_with_entity_column_retry(sql, pair, pair.d0, registry)
    out1 = _execute_sql_with_entity_column_retry(sql, pair, pair.d1, registry)
    value0 = _observed_output(out0, pair.measure.measure_id)
    value1 = _observed_output(out1, pair.measure.measure_id)
    c0 = contract_value(pair, pair.d0)
    c1 = contract_value(pair, pair.d1)
    pair_preserved = (c0 == c1) if contract_preserved is None else contract_preserved
    output_changed = not outputs_equal(value0, value1, mode=output_comparator)
    violated = (pair_preserved and output_changed) if require_contract_preservation else output_changed
    return ValidationResult(
        status="VIOLATION" if violated else "PASS",
        test_type=pair.test_type,
        measure=pair.measure.measure_id,
        pass_class=None if violated else "EXECUTABLE_PASS",
        contract_value_d0=c0,
        contract_value_d1=c1,
        sql_output_d0=value0,
        sql_output_d1=value1,
        diagnosis=_diagnosis(pair, require_contract_preservation=require_contract_preservation)
        if violated
        else "no invariant violation observed",
        contract_preserved=pair_preserved if not require_contract_preservation else None,
    )


def outputs_equal(left: Any, right: Any, *, mode: str = "exact") -> bool:
    """Compare projected outputs under a declared sensitivity-analysis policy."""
    if mode == "exact":
        return left == right
    if mode == "order_normalized":
        return _order_normalized(left) == _order_normalized(right)
    if mode == "numeric_tolerant":
        return _numeric_tolerant_equal(left, right)
    raise ValueError(f"unsupported output comparator: {mode}")


def _order_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _order_normalized(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, list):
        items = [_order_normalized(item) for item in value]
        return tuple(sorted(items, key=repr))
    return value


def _numeric_tolerant_equal(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(_numeric_tolerant_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        unmatched = list(right)
        for left_item in left:
            match = next(
                (
                    index
                    for index, right_item in enumerate(unmatched)
                    if _numeric_tolerant_equal(left_item, right_item)
                ),
                None,
            )
            if match is None:
                return False
            unmatched.pop(match)
        return True
    return left == right


def _is_number(value: Any) -> bool:
    return isinstance(value, (Real, Decimal)) and not isinstance(value, bool)


def _execute_sql_with_entity_column_retry(
    sql: str,
    pair: TestPair,
    rows: dict[str, list[dict[str, Any]]],
    registry: SchemaRegistry,
) -> list[dict[str, Any]]:
    try:
        return execute_sql(sql, rows, registry)
    except sqlite3.OperationalError as exc:
        message = str(exc)
        prefix = "ambiguous column name:"
        if prefix not in message.lower():
            raise
        rewritten = _qualify_ambiguous_entity_column(sql, pair)
        if rewritten == sql:
            raise
        return execute_sql(rewritten, rows, registry)


def _qualify_ambiguous_entity_column(sql: str, pair: TestPair) -> str:
    entity_key = pair.measure.entity_key
    if not entity_key or "." not in entity_key:
        return sql
    entity_table, entity_col = entity_key.split(".", 1)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return sql

    changed = False

    def qualify(node: exp.Expression) -> exp.Expression:
        nonlocal changed
        if isinstance(node, exp.Column) and not node.table and node.name.lower() == entity_col.lower():
            changed = True
            return exp.column(entity_col, table=entity_table)
        return node

    rewritten = tree.transform(qualify).sql(dialect="sqlite")
    return rewritten if changed else sql


def _observed_output(rows: list[dict[str, Any]], measure_id: str) -> Any:
    if not rows:
        return None
    row = rows[0]
    if measure_id in row:
        return row[measure_id]
    if "ranking_metric" in row:
        return row["ranking_metric"]
    numeric_values = [
        value
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if len(rows) == 1 and len(numeric_values) == 1:
        return numeric_values[0]
    return rows


def _diagnosis(pair: TestPair, *, require_contract_preservation: bool = True) -> str:
    if not require_contract_preservation:
        return "SQL output changed without requiring contract truth preservation"
    if pair.test_type == "CRT":
        return f"{pair.measure.source_table}-level measure is duplicated after joining {pair.meta.get('child_table')}"
    if pair.test_type == "GRT":
        return f"{pair.measure.source_table} grain measure is duplicated after joining {pair.meta.get('child_table')}"
    if pair.test_type == "CGR":
        return f"{pair.measure.source_table} grain count is duplicated after joining {pair.meta.get('child_table')}"
    if pair.test_type == "DET":
        return "entity count is sensitive to duplicate event rows; COUNT DISTINCT is likely required"
    if pair.test_type == "EST":
        return "entity-level average is sensitive to event splitting; two-stage aggregation is likely required"
    if pair.test_type == "ART":
        return "average is reweighted by non-uniform join replication; average once per base grain is likely required"
    if pair.test_type == "ELT":
        return "entity-list output is sensitive to duplicate detail rows; DISTINCT or entity-key grouping is likely required"
    return "aggregation grain invariant violation"
