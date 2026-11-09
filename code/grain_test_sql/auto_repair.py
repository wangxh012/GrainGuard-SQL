from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

from .contract import MetricContract, MeasureContract
from .schema_registry import SchemaRegistry


@dataclass(frozen=True)
class RepairAttempt:
    sql: str | None
    strategy: str
    reason: str
    structured_hint: dict[str, Any]


def build_structured_repair_hint(
    contract: MetricContract,
    grain_result: dict[str, Any],
    registry: SchemaRegistry | None = None,
) -> dict[str, Any]:
    violations = [result for result in grain_result.get("results", []) if result.get("status") == "VIOLATION"]
    test_types = sorted({result.get("test_type") for result in violations if result.get("test_type")})
    measures = []
    by_id = {measure.measure_id: measure for measure in contract.measures}
    for result in violations:
        measure = by_id.get(result.get("measure"))
        if not measure:
            continue
        measures.append(_measure_hint(measure, result, registry))
    return {
        "status": grain_result.get("status"),
        "violation_test_types": test_types,
        "output_group": contract.output_group.ref,
        "measures": measures,
        "repair_policy": _repair_policy(test_types),
    }


def deterministic_repair(
    sql: str,
    contract: MetricContract,
    grain_result: dict[str, Any],
    registry: SchemaRegistry,
) -> RepairAttempt:
    hint = build_structured_repair_hint(contract, grain_result, registry)
    if grain_result.get("status") != "VIOLATION":
        return RepairAttempt(None, "none", "grain validation did not report a violation", hint)

    violations = [result for result in grain_result.get("results", []) if result.get("status") == "VIOLATION"]
    measures = {measure.measure_id: measure for measure in contract.measures}
    for violation in violations:
        measure = measures.get(violation.get("measure"))
        test_type = violation.get("test_type")
        if not measure:
            continue
        if test_type == "DET":
            repaired = _repair_det_conditional_ratio(sql, measure, registry)
            if repaired:
                return RepairAttempt(
                    repaired,
                    "deterministic_det_distinct_conditional_ratio",
                    "rewrote conditional ratio to count distinct entities in numerator and denominator",
                    hint,
                )
            repaired = _repair_count_distinct(sql, _det_distinct_ref(measure, registry), registry)
            if repaired:
                return RepairAttempt(repaired, "deterministic_det_count_distinct", "rewrote COUNT to COUNT(DISTINCT entity key)", hint)
            repaired = _repair_distinct_projection(sql)
            if repaired:
                return RepairAttempt(
                    repaired,
                    "deterministic_det_distinct_projection",
                    "added DISTINCT to a duplicated detail-row projection",
                    hint,
                )
        if test_type == "CGR":
            repaired = _repair_count_distinct(sql, _single_base_grain_ref(measure), registry)
            if repaired:
                return RepairAttempt(repaired, "deterministic_cgr_count_distinct", "rewrote COUNT to COUNT(DISTINCT base grain)", hint)
        if test_type == "ART":
            repaired = _repair_avg_once_per_grain(sql, measure, registry)
            if repaired:
                return RepairAttempt(
                    repaired,
                    "deterministic_art_distinct_grain_avg",
                    "averaged over a DISTINCT base-grain subquery before lower-grain joins can reweight rows",
                    hint,
                )
        if test_type == "EST":
            repaired = _repair_avg_entity_total(sql, measure, registry)
            if repaired:
                return RepairAttempt(
                    repaired,
                    "deterministic_est_entity_total_avg",
                    "aggregated event values by entity before averaging entity-level totals",
                    hint,
                )

    return RepairAttempt(None, "llm_required", "no safe deterministic rewrite matched this SQL shape", hint)


def _measure_hint(
    measure: MeasureContract,
    result: dict[str, Any],
    registry: SchemaRegistry | None,
) -> dict[str, Any]:
    hint: dict[str, Any] = {
        "measure_id": measure.measure_id,
        "test_type": result.get("test_type"),
        "semantic_formula": measure.semantic_formula,
        "source_table": measure.source_table,
        "source_column": measure.source_column,
        "base_grain": measure.base_grain,
        "entity_key": measure.entity_key,
        "event_table": measure.event_table,
        "diagnosis": result.get("diagnosis"),
        "contract_value_D0": result.get("contract_value_D0"),
        "contract_value_D1": result.get("contract_value_D1"),
        "sql_output_D0": result.get("sql_output_D0"),
        "sql_output_D1": result.get("sql_output_D1"),
    }
    if registry and measure.entity_key and measure.event_table:
        entity_table, _ = measure.entity_key.split(".", 1)
        fk = registry.fk_to_parent(measure.event_table, entity_table)
        if fk:
            hint["event_to_entity_fk"] = {
                "event_table": measure.event_table,
                "event_columns": list(fk.columns),
                "entity_table": entity_table,
                "entity_columns": list(fk.ref_columns),
            }
    return hint


def _repair_policy(test_types: list[str]) -> list[str]:
    policies = []
    if "DET" in test_types:
        policies.append("Use COUNT(DISTINCT entity_key) or an EXISTS semi-join so event rows cannot duplicate entities.")
    if "CGR" in test_types:
        policies.append("Count distinct base_grain keys or de-duplicate base_grain rows before lower-grain joins.")
    if "CRT" in test_types or "GRT" in test_types:
        policies.append("Pre-aggregate source-table measures at their base grain before joining lower-grain tables.")
    if "ART" in test_types:
        policies.append("Average one value per base grain; avoid letting non-uniform join replication reweight rows.")
    if "EST" in test_types:
        policies.append("Aggregate event values by entity first, then average the entity-level totals.")
    return policies


def _det_distinct_ref(measure: MeasureContract, registry: SchemaRegistry) -> str | None:
    if not measure.entity_key:
        return None
    entity_table, entity_col = measure.entity_key.split(".", 1)
    return measure.entity_key if entity_col and entity_table in registry.tables else None


def _single_base_grain_ref(measure: MeasureContract) -> str | None:
    if len(measure.base_grain) != 1:
        return None
    return measure.base_grain[0]


def _repair_count_distinct(sql: str, target_ref: str | None, registry: SchemaRegistry) -> str | None:
    if not target_ref or "." not in target_ref:
        return None
    target_table, target_col = target_ref.split(".", 1)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None

    aliases = _table_aliases(tree)
    target_column = _distinct_column_for_ref(target_table, target_col, aliases, registry)
    if target_column is None:
        return None

    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if not select:
        return None
    count_expr = _first_top_level_count(select)
    if not count_expr:
        return None

    count_expr.set("this", exp.Distinct(expressions=[target_column]))
    return tree.sql(dialect="sqlite")


def _repair_det_conditional_ratio(sql: str, measure: MeasureContract, registry: SchemaRegistry) -> str | None:
    if not measure.entity_key or "." not in measure.entity_key:
        return None
    target_table, target_col = measure.entity_key.split(".", 1)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if not select or select.args.get("group") or select.args.get("having"):
        return None

    aliases = _table_aliases(tree)
    distinct_column = _distinct_column_for_ref(target_table, target_col, aliases, registry)
    if distinct_column is None:
        return None

    replaced_sum = False
    replaced_count = False
    for projection in select.expressions:
        for node in list(projection.walk()):
            if isinstance(node, exp.Sum) and isinstance(node.this, exp.Case):
                condition = _single_case_condition(node.this)
                if condition is None:
                    continue
                counted_case = exp.Case(
                    ifs=[exp.If(this=condition.copy(), true=distinct_column.copy())],
                )
                node.replace(exp.Count(this=exp.Distinct(expressions=[counted_case])))
                replaced_sum = True
            elif isinstance(node, exp.Count):
                node.set("this", exp.Distinct(expressions=[distinct_column.copy()]))
                replaced_count = True

    return tree.sql(dialect="sqlite") if replaced_sum and replaced_count else None


def _repair_distinct_projection(sql: str) -> str | None:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if not select or select.args.get("distinct"):
        return None
    for projection in select.expressions:
        if _has_aggregate(projection):
            return None
    select.set("distinct", exp.Distinct())
    return tree.sql(dialect="sqlite")


def _repair_avg_once_per_grain(sql: str, measure: MeasureContract, registry: SchemaRegistry) -> str | None:
    if not measure.source_column or not measure.base_grain:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if not select or select.args.get("having"):
        return None

    avg_expr = _first_top_level_avg(select)
    if not avg_expr:
        return None
    aliases = _table_aliases(tree)
    source_alias = aliases.get(measure.source_table)
    if not source_alias:
        return None
    avg_source = _first_column_ref(avg_expr)
    if avg_source and not _same_sql_ref(avg_source, f"{source_alias}.{measure.source_column}"):
        return None

    group = select.args.get("group")
    group_expressions = list(group.expressions) if group else []
    subquery = select.copy()
    expressions: list[exp.Expression] = []
    for index, group_expr in enumerate(group_expressions):
        expressions.append(exp.alias_(group_expr.copy(), f"__group_{index}"))
    for index, grain_ref in enumerate(measure.base_grain):
        column = _column_for_contract_ref(grain_ref, aliases, registry)
        if column is None:
            return None
        expressions.append(exp.alias_(column, f"__grain_{index}"))
    expressions.append(exp.alias_(exp.column(measure.source_column, table=source_alias), "__measure_value"))
    subquery.set("expressions", expressions)
    subquery.set("distinct", exp.Distinct())
    for key in ("order", "limit"):
        subquery.set(key, None)

    projection_alias = _projection_alias_for_aggregate(select, avg_expr)
    outer_projection = "AVG(__measure_value)"
    if projection_alias:
        outer_projection = f"{outer_projection} AS {_sqlite_identifier(projection_alias)}"
    if not group_expressions:
        return f"SELECT {outer_projection} FROM ({subquery.sql(dialect='sqlite')}) AS grain_base"

    group_selects = []
    group_by = []
    for index, group_expr in enumerate(group_expressions):
        alias = f"__group_{index}"
        display_alias = _group_display_alias(group_expr)
        group_selects.append(f"{alias} AS {_sqlite_identifier(display_alias)}" if display_alias else alias)
        group_by.append(alias)
    return (
        f"SELECT {', '.join(group_selects + [outer_projection])} "
        f"FROM ({subquery.sql(dialect='sqlite')}) AS grain_base "
        f"GROUP BY {', '.join(group_by)}"
    )


def _repair_avg_entity_total(sql: str, measure: MeasureContract, registry: SchemaRegistry) -> str | None:
    if not measure.source_column or not measure.source_table or not measure.entity_key:
        return None
    if "." not in measure.entity_key:
        return None
    target_table, target_col = measure.entity_key.split(".", 1)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if not select or select.args.get("having"):
        return None

    avg_expr = _first_top_level_avg(select)
    if not avg_expr:
        return None
    aliases = _table_aliases(tree)
    source_alias = aliases.get(measure.source_table)
    if not source_alias:
        return None
    avg_source = _first_column_ref(avg_expr)
    if avg_source and not _same_sql_ref(avg_source, f"{source_alias}.{measure.source_column}"):
        return None
    entity_column = _distinct_column_for_ref(target_table, target_col, aliases, registry)
    if entity_column is None:
        return None

    group = select.args.get("group")
    group_expressions = list(group.expressions) if group else []
    subquery = select.copy()
    expressions: list[exp.Expression] = []
    for index, group_expr in enumerate(group_expressions):
        expressions.append(exp.alias_(group_expr.copy(), f"__group_{index}"))
    expressions.append(exp.alias_(entity_column.copy(), "__entity_key"))
    expressions.append(
        exp.alias_(
            exp.Sum(this=exp.column(measure.source_column, table=source_alias)),
            "__entity_total",
        )
    )
    subquery.set("expressions", expressions)
    subquery.set("group", exp.Group(expressions=[*(expr.copy() for expr in group_expressions), entity_column.copy()]))
    for key in ("order", "limit"):
        subquery.set(key, None)

    projection_alias = _projection_alias_for_aggregate(select, avg_expr)
    outer_projection = "AVG(__entity_total)"
    if projection_alias:
        outer_projection = f"{outer_projection} AS {_sqlite_identifier(projection_alias)}"
    if not group_expressions:
        return f"SELECT {outer_projection} FROM ({subquery.sql(dialect='sqlite')}) AS entity_base"

    group_selects = []
    group_by = []
    for index, group_expr in enumerate(group_expressions):
        alias = f"__group_{index}"
        display_alias = _group_display_alias(group_expr)
        group_selects.append(f"{alias} AS {_sqlite_identifier(display_alias)}" if display_alias else alias)
        group_by.append(alias)
    return (
        f"SELECT {', '.join(group_selects + [outer_projection])} "
        f"FROM ({subquery.sql(dialect='sqlite')}) AS entity_base "
        f"GROUP BY {', '.join(group_by)}"
    )


def _table_aliases(tree: exp.Expression) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        aliases.setdefault(table.name, table.alias_or_name)
    return aliases


def _column_for_contract_ref(
    ref: str,
    aliases: dict[str, str],
    registry: SchemaRegistry,
) -> exp.Column | None:
    if "." not in ref:
        return None
    table, column = ref.split(".", 1)
    alias = aliases.get(table) or _compatible_alias(table, column, aliases, registry)
    if not alias:
        return None
    return exp.column(column, table=alias)


def _compatible_alias(
    target_table: str,
    target_col: str,
    aliases: dict[str, str],
    registry: SchemaRegistry,
) -> str | None:
    if target_table not in registry.tables:
        return None
    for table_name, alias in aliases.items():
        if table_name not in registry.tables:
            continue
        fk = registry.fk_to_parent(table_name, target_table)
        if fk and len(fk.columns) == 1 and fk.ref_columns == [target_col]:
            return alias
    for table_name, alias in aliases.items():
        if table_name in registry.tables and target_col in registry.tables[table_name].columns:
            return alias
    return None


def _distinct_column_for_ref(
    target_table: str,
    target_col: str,
    aliases: dict[str, str],
    registry: SchemaRegistry,
) -> exp.Column | None:
    target_alias = aliases.get(target_table)
    if target_alias:
        return exp.column(target_col, table=target_alias)
    if target_table not in registry.tables:
        return None
    for table_name, alias in aliases.items():
        if table_name not in registry.tables:
            continue
        fk = registry.fk_to_parent(table_name, target_table)
        if fk and len(fk.columns) == 1:
            return exp.column(fk.columns[0], table=alias)
    for table_name, alias in aliases.items():
        if table_name in registry.tables and target_col in registry.tables[table_name].columns:
            return exp.column(target_col, table=alias)
    return None


def _first_top_level_count(select: exp.Select) -> exp.Count | None:
    for projection in select.expressions:
        for node in projection.walk():
            if isinstance(node, exp.Count):
                return node
    return None


def _first_top_level_avg(select: exp.Select) -> exp.Avg | None:
    for projection in select.expressions:
        for node in projection.walk():
            if isinstance(node, exp.Avg):
                return node
    return None


def _first_column_ref(node: exp.Expression) -> str | None:
    for column in node.find_all(exp.Column):
        if column.table:
            return f"{column.table}.{column.name}"
        return column.name
    return None


def _same_sql_ref(left: str, right: str) -> bool:
    return left.lower() == right.lower()


def _single_case_condition(case: exp.Case) -> exp.Expression | None:
    ifs = case.args.get("ifs") or []
    if len(ifs) != 1:
        return None
    return ifs[0].args.get("this")


def _has_aggregate(node: exp.Expression) -> bool:
    aggregate_types = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)
    return any(isinstance(child, aggregate_types) for child in node.walk())


def _projection_alias_for_aggregate(select: exp.Select, aggregate: exp.Expression) -> str | None:
    for projection in select.expressions:
        if aggregate in list(projection.walk()):
            return projection.alias
    return None


def _group_display_alias(group_expr: exp.Expression) -> str | None:
    if isinstance(group_expr, exp.Column):
        return group_expr.name
    return None


def _sqlite_identifier(name: str) -> str:
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        return name
    return '"' + name.replace('"', '""') + '"'
