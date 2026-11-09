from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from .contract import MeasureContract
from .schema_registry import SchemaRegistry
from .validator import ValidationResult


def repair_hint(
    sql: str,
    result: ValidationResult,
    measure: MeasureContract | None = None,
    registry: SchemaRegistry | None = None,
) -> str:
    if result.test_type == "DET":
        return _count_distinct_hint(sql, _det_distinct_ref(sql, measure, registry))
    if result.test_type in {"CRT", "GRT"}:
        return "Repair hint: pre-aggregate the parent-grain measure before joining lower-grain child tables."
    if result.test_type == "CGR":
        return "Repair hint: count distinct base-grain keys or de-duplicate base-grain rows before joining replicated tables."
    if result.test_type == "ART":
        return "Repair hint: pre-aggregate or de-duplicate the averaged base-grain records before joining replicated tables."
    if result.test_type == "EST":
        return "Repair hint: first aggregate event values by entity, then AVG the entity totals by output group."
    return "Repair hint unavailable."


def _count_distinct_hint(sql: str, target_ref: str | None = None) -> str:
    if target_ref:
        repaired = _replace_first_count_with_distinct(sql, target_ref)
        if repaired and repaired != sql:
            return f"Suggested SQL:\n{repaired}"
        return f"Repair hint: replace the counted expression with COUNT(DISTINCT {target_ref})."
    if re.search(r"COUNT\s*\(\s*\*\s*\)", sql, re.IGNORECASE):
        return "Repair hint: replace COUNT(*) with COUNT(DISTINCT entity key) for the counted entity."
    pattern = re.compile(r"COUNT\s*\(\s*([A-Za-z_][\w.]*?)\s*\)", re.IGNORECASE)
    repaired = pattern.sub(r"COUNT(DISTINCT \1)", sql, count=1)
    if repaired != sql:
        return f"Suggested SQL:\n{repaired}"
    return "Repair hint: replace COUNT(entity_key) with COUNT(DISTINCT entity_key)."


def _det_distinct_ref(
    sql: str,
    measure: MeasureContract | None,
    registry: SchemaRegistry | None,
) -> str | None:
    if not measure or not measure.entity_key or "." not in measure.entity_key:
        return None
    entity_table, entity_col = measure.entity_key.split(".", 1)
    aliases = _table_aliases(sql)
    if entity_table in aliases:
        return f"{aliases[entity_table]}.{entity_col}"
    if registry and measure.event_table:
        fk = registry.fk_to_parent(measure.event_table, entity_table)
        if fk and len(fk.columns) == 1 and measure.event_table in aliases:
            return f"{aliases[measure.event_table]}.{fk.columns[0]}"
    return measure.entity_key


def _replace_first_count_with_distinct(sql: str, target_ref: str) -> str | None:
    if "." not in target_ref:
        return None
    table, column = target_ref.split(".", 1)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if not select:
        return None
    count_expr = next((node for node in select.expressions for node in node.find_all(exp.Count)), None)
    if not count_expr:
        return None
    count_expr.set("this", exp.Distinct(expressions=[exp.column(column, table=table)]))
    return tree.sql(dialect="sqlite")


def _table_aliases(sql: str) -> dict[str, str]:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return {}
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        aliases.setdefault(table.name, table.alias_or_name)
    return aliases
