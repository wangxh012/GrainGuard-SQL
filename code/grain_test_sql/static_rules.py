from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from .contract import MetricContract
from .schema_registry import SchemaRegistry
from .signature_extractor import SQLSignature, extract_signature
from .template_matcher import (
    _det_row_count_replication_tables,
    _has_agg_on_source,
    _has_count_on_grain,
    _has_count_without_distinct,
    _joined_replication_tables,
    _needs_sum_split_witness,
    _query_join_replication_paths,
)


@dataclass(frozen=True)
class StaticRuleDecision:
    status: str
    reason: str

    @property
    def is_alert(self) -> bool:
        return self.status == "ALERT"


def hard_static_grain_decision(
    sql: str,
    registry: SchemaRegistry,
    contract: MetricContract,
) -> StaticRuleDecision:
    """High-confidence static grain-risk rules.

    This baseline intentionally stops before witness construction/execution. It
    alerts only when the SQL signature itself matches a known grain anti-pattern.
    """
    try:
        signature = extract_signature(sql)
    except Exception as exc:
        return StaticRuleDecision("UNSUPPORTED", f"SQL signature extraction failed: {exc}")

    reasons: list[str] = []
    for measure in contract.measures:
        reason = _measure_alert_reason(measure, signature, registry, contract.output_group.ref)
        if reason:
            reasons.append(f"{measure.measure_id}: {reason}")

    if reasons:
        return StaticRuleDecision("ALERT", "; ".join(reasons))
    return StaticRuleDecision("PASS", "no high-confidence static grain anti-pattern matched")


def _measure_alert_reason(
    measure,
    signature: SQLSignature,
    registry: SchemaRegistry,
    output_group_ref: str,
) -> str | None:
    requested = set(measure.applicable_tests)

    if signature.synthetic_fault:
        return "controlled synthetic grain-fault pattern is present"

    non_unique_group = _non_unique_entity_group_reason(measure, signature, registry, output_group_ref)
    if non_unique_group:
        return non_unique_group

    duplicate_entity_list = _duplicating_entity_list_reason(measure, signature, registry, output_group_ref)
    if duplicate_entity_list:
        return duplicate_entity_list

    fk_literal_filter = _foreign_key_literal_filter_reason(measure, signature, registry)
    if fk_literal_filter:
        return fk_literal_filter

    missing_zero_group = _missing_zero_group_reason(measure, signature, registry)
    if missing_zero_group:
        return missing_zero_group

    if requested.intersection({"CRT", "GRT"}) and measure.semantic_formula == "SUM_ONCE_PER_GRAIN":
        if _has_agg_on_source(signature, "SUM", measure.source_ref, registry):
            child = _unsafe_joined_replication_child(measure.source_table, signature, registry)
            if child:
                return f"SUM over {measure.source_ref} after non-unique join to {child}"
        if _needs_sum_split_witness(measure, signature, registry):
            if _has_agg_on_source(signature, "SUM", measure.source_ref, registry):
                return "SUM metric is mixed with a row COUNT at a different unit"
            return (
                f"SUM contract reads {measure.source_ref} at detail-row grain "
                "without aggregating it"
            )

    if requested.intersection({"CGR", "GRT"}) and measure.semantic_formula == "COUNT_ONCE_PER_GRAIN":
        if _has_count_on_grain(signature, measure, registry):
            child = _unsafe_joined_replication_child(measure.source_table, signature, registry)
            if child:
                return f"COUNT over base grain after non-unique join to {child}"

    if "DET" in requested and measure.semantic_formula == "COUNT_DISTINCT_ENTITY":
        non_distinct_attribute = _non_distinct_entity_attribute_count_reason(measure, signature, registry)
        if non_distinct_attribute:
            return non_distinct_attribute
        if _has_count_without_distinct(signature, measure, registry):
            if _event_can_duplicate_entity(measure, registry, signature):
                return (
                    "entity count uses a row-counting aggregate without DISTINCT "
                    "over a duplicating event table"
                )

    if "EST" in requested and measure.semantic_formula in {"AVG_ENTITY_TOTAL", "AVG_ONCE_PER_ENTITY"}:
        if _has_agg_on_source(signature, "AVG", measure.source_ref, registry):
            if _event_can_duplicate_entity(measure, registry, signature):
                return "entity-level average is computed directly over event/detail rows"

    if "ART" in requested and measure.semantic_formula == "AVG_ONCE_PER_GRAIN":
        if _has_agg_on_source(signature, "AVG", measure.source_ref, registry):
            child = _unsafe_joined_replication_child(measure.source_table, signature, registry)
            if child:
                return f"AVG over {measure.source_ref} after non-uniform join to {child}"

    if "TST" in requested and measure.semantic_formula == "TOPK_BY_METRIC":
        detail_row_topk = _detail_row_topk_reason(
            measure,
            signature,
            registry,
            output_group_ref,
        )
        if detail_row_topk:
            return detail_row_topk
        bare_group_topk = _bare_group_topk_reason(measure, signature, registry)
        if bare_group_topk:
            return bare_group_topk
        if signature.has_order_limit and any(m.function in {"SUM", "COUNT", "AVG"} for m in signature.measures):
            child = _unsafe_joined_replication_child(measure.source_table, signature, registry)
            if child:
                return f"Top-k aggregate can be changed by non-unique join to {child}"

    return None


def _detail_row_topk_reason(
    measure,
    signature: SQLSignature,
    registry: SchemaRegistry,
    output_group_ref: str,
) -> str | None:
    """Detect ranking detail rows when the contract targets a coarser group."""
    if signature.groups or not measure.source_ref:
        return None
    source = _resolve_ref(measure.source_ref, registry, signature)
    output = _resolve_ref(output_group_ref, registry, signature)
    if not source or not output:
        return None
    source_table, source_column = source
    output_table, output_column = output
    if source_table != output_table:
        return None
    if registry.columns_are_unique(output_table, [output_column]):
        return None
    if _ranking_group_is_single_row(
        measure,
        signature,
        registry,
        output,
    ):
        return None
    detail_order = signature.has_order_limit and any(
        _same_resolved_ref(
            ordered,
            measure.source_ref,
            registry,
            signature,
        )
        for ordered in signature.order_columns
    )
    scalar_extreme = any(
        sql_measure.function in {"MAX", "MIN"}
        and _same_resolved_ref(
            sql_measure.source,
            measure.source_ref,
            registry,
            signature,
        )
        for sql_measure in signature.measures
    )
    if not detail_order and not scalar_extreme:
        return None
    if not scalar_extreme and not any(
        _refs_identify_same_grain(
            projection,
            output_group_ref,
            registry,
            signature,
        )
        for projection in signature.projections
    ):
        return None
    if not scalar_extreme and any(
        _same_resolved_ref(
            sql_measure.source,
            measure.source_ref,
            registry,
            signature,
        )
        for sql_measure in signature.measures
    ):
        return None
    grain_refs = [
        resolved
        for grain in measure.base_grain
        if (resolved := _resolve_ref(grain, registry, signature))
    ]
    if (
        measure.ranking_metric_semantic != "SUM_ONCE_PER_GRAIN"
        or output not in grain_refs
        or len(set(grain_refs)) < 2
    ):
        return None
    return (
        f"TOPK orders detail-row metric {source_table}.{source_column} without "
        f"aggregating it per target unit {output_table}.{output_column}"
    )


def _ranking_group_is_single_row(
    measure,
    signature: SQLSignature,
    registry: SchemaRegistry,
    output: tuple[str, str],
) -> bool:
    """Whether exact predicates fix every base-grain dimension besides group."""
    output_table, output_column = output
    if output_table != measure.source_table:
        return False
    remaining = {
        resolved
        for grain in measure.base_grain
        if (resolved := _resolve_ref(grain, registry, signature))
        and resolved != (output_table, output_column)
    }
    if not remaining:
        return True
    constrained = {
        resolved
        for item in signature.filters
        if item.operator in {"=", "EQ"}
        and (resolved := _resolve_ref(item.source, registry, signature))
    }
    return remaining <= constrained


def _unsafe_joined_replication_child(
    source_table: str,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> str | None:
    for child, schema_safe in _joined_replication_tables(source_table, signature, registry):
        if not schema_safe:
            return child
    paths = _query_join_replication_paths(source_table, signature, registry)
    if paths:
        return paths[0][0]
    return None


def _foreign_key_literal_filter_reason(measure, signature: SQLSignature, registry: SchemaRegistry) -> str | None:
    for sql_filter in signature.filters:
        if not isinstance(sql_filter.value, str):
            continue
        resolved = _resolve_ref(sql_filter.source, registry, signature)
        if not resolved:
            continue
        table, column = resolved
        if table != measure.source_table:
            continue
        fk = next((fk for fk in registry.tables[table].foreign_keys if column in fk.columns), None)
        if not fk:
            continue
        if _looks_like_identifier_literal(sql_filter.value):
            continue
        return (
            f"filter compares foreign-key column {table}.{column} to text literal "
            f"{sql_filter.value!r}; join/filter the referenced {fk.ref_table} entity instead"
        )
    return None


def _missing_zero_group_reason(measure, signature: SQLSignature, registry: SchemaRegistry) -> str | None:
    if measure.semantic_formula != "COUNT_ONCE_PER_GRAIN":
        return None
    try:
        tree = sqlglot.parse_one(signature.sql, read="sqlite")  # type: ignore[attr-defined]
    except Exception:
        return None

    aliases = _table_aliases(tree)
    for having in tree.find_all(exp.Having):
        comparison = having.find(exp.LT) or having.find(exp.LTE)
        if not comparison or not _positive_numeric_literal(comparison.right):
            continue
        count = comparison.left.find(exp.Count) if comparison.left else None
        if not count:
            continue
        count_source = _first_column_table(count, aliases)
        if not count_source or count_source == measure.source_table:
            continue
        fk = registry.fk_to_parent(count_source, measure.source_table) if count_source in registry.tables else None
        if not fk or not any(group.startswith(f"{measure.source_table}.") for group in signature.groups):
            continue
        if _has_left_join_to_table(tree, count_source, aliases):
            continue
        return (
            f"HAVING COUNT over {count_source} after an inner join to {measure.source_table} "
            "cannot return zero-child groups"
        )
    return None


def _non_distinct_entity_attribute_count_reason(
    measure,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> str | None:
    if not measure.entity_key:
        return None
    entity_table, entity_col = measure.entity_key.split(".", 1)
    entity_table = registry.resolve_table_name(entity_table)
    entity_col = registry.resolve_column_name(entity_table, entity_col)
    for sql_measure in signature.measures:
        if sql_measure.function != "COUNT" or sql_measure.distinct or not sql_measure.source:
            continue
        resolved = _resolve_ref(sql_measure.source, registry, signature)
        if not resolved:
            continue
        table, column = resolved
        if table != entity_table or column == entity_col:
            continue
        if registry.columns_are_unique(entity_table, [column]):
            continue
        return (
            f"COUNT over non-unique {entity_table}.{column} is not a distinct "
            f"{entity_table}.{entity_col} count"
        )
    return None


def _bare_group_topk_reason(measure, signature: SQLSignature, registry: SchemaRegistry) -> str | None:
    if not measure.source_ref or not signature.groups:
        return None
    source_resolved = _resolve_ref(measure.source_ref, registry, signature)
    if not source_resolved:
        return None
    source_ref = f"{source_resolved[0]}.{source_resolved[1]}"
    grouped_grain = any(
        _refs_identify_same_grain(group, grain, registry, signature)
        for group in signature.groups
        for grain in measure.base_grain
    )
    if not grouped_grain:
        return None
    if any(m.function == "MAX" and _same_resolved_ref(m.source, source_ref, registry, signature) for m in signature.measures):
        return None
    if any(_same_resolved_ref(m.source, source_ref, registry, signature) for m in signature.measures):
        return None
    if not any(_same_resolved_ref(proj, measure.source_ref, registry, signature) for proj in signature.projections):
        return (
            f"TOPK metric {source_ref} is ordered around grouped rows without computing "
            "the per-grain maximum"
        )
    return None


def _event_can_duplicate_entity(measure, registry: SchemaRegistry, signature: SQLSignature) -> bool:
    if not measure.entity_key:
        return False
    entity_table, entity_col = measure.entity_key.split(".", 1)
    event_tables = _det_row_count_replication_tables(
        signature,
        measure,
        registry,
    )
    if not event_tables and measure.event_table:
        event_tables = [measure.event_table]
    return any(
        not registry.child_is_unique_for_parent_under_filters(
            event_table,
            entity_table,
            signature.filters,
            registry.fk_to_parent(
                event_table,
                entity_table,
                ref_column=entity_col,
            ),
        )
        for event_table in event_tables
    )


def _non_unique_entity_group_reason(
    measure,
    signature: SQLSignature,
    registry: SchemaRegistry,
    output_group_ref: str,
) -> str | None:
    if not signature.groups:
        return None
    output = _resolve_ref(output_group_ref, registry)
    if not output:
        return None
    output_table, _ = output
    if not _measure_grain_points_to_entity(measure, output_table, registry):
        return None
    for group_ref in signature.groups:
        resolved = _resolve_ref(group_ref, registry)
        if not resolved:
            continue
        group_table, group_col = resolved
        if group_table != output_table:
            continue
        if registry.columns_are_known_non_unique(group_table, [group_col]):
            return (
                f"GROUP BY {group_table}.{group_col} merges multiple {output_table} entities; "
                "group by the entity key before counting"
            )
    return None


def _duplicating_entity_list_reason(
    measure,
    signature: SQLSignature,
    registry: SchemaRegistry,
    output_group_ref: str,
) -> str | None:
    if signature.measures or signature.groups or signature.distinct:
        return None
    entity_table = _entity_table_for_measure(measure, output_group_ref, registry)
    relation_tables = signature.outer_tables or signature.tables
    if not entity_table or entity_table not in relation_tables:
        return None
    if not any(
        _projection_table(ref, registry, signature) == entity_table
        for ref in signature.projections
    ):
        return None
    for child, schema_safe in _joined_replication_tables(entity_table, signature, registry):
        # If the outer result explicitly projects a child/detail attribute, its
        # target unit is at least (entity, detail). Repeated entity values are
        # then expected and are not evidence of an entity-list grain defect.
        if any(
            _projection_table(ref, registry, signature) == child
            for ref in signature.projections
        ):
            continue
        if child in relation_tables and not schema_safe:
            return (
                f"entity-list query projects {entity_table} rows after joining duplicating child table {child} "
                "without DISTINCT or GROUP BY"
            )
    return None


def _measure_grain_points_to_entity(measure, entity_table: str, registry: SchemaRegistry) -> bool:
    for grain_ref in measure.base_grain:
        resolved = _resolve_ref(grain_ref, registry)
        if not resolved:
            continue
        table, col = resolved
        if table == entity_table and registry.columns_are_unique(table, [col]):
            return True
        if table in registry.tables:
            if any(
                fk.ref_table == entity_table and col in fk.columns
                for fk in registry.tables[table].foreign_keys
            ):
                return True
    if measure.entity_key:
        resolved = _resolve_ref(measure.entity_key, registry)
        if resolved and resolved[0] == entity_table:
            return True
    return False


def _entity_table_for_measure(measure, output_group_ref: str, registry: SchemaRegistry) -> str | None:
    if measure.entity_key:
        resolved = _resolve_ref(measure.entity_key, registry)
        if resolved:
            return resolved[0]
    output = _resolve_ref(output_group_ref, registry)
    if output:
        return output[0]
    if measure.source_table in registry.tables:
        return measure.source_table
    return None


def _projection_table(
    ref: str,
    registry: SchemaRegistry,
    signature: SQLSignature | None = None,
) -> str | None:
    resolved = _resolve_ref(ref, registry, signature)
    return resolved[0] if resolved else None


def _looks_like_identifier_literal(value: str) -> bool:
    return bool(value) and (value.isdigit() or value.isupper() or any(char.isdigit() for char in value))


def _table_aliases(tree: exp.Expression) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        aliases[table.alias_or_name] = table.name
        aliases[table.name] = table.name
    return aliases


def _positive_numeric_literal(node: exp.Expression | None) -> bool:
    if not isinstance(node, exp.Literal):
        return False
    try:
        return float(node.this) > 0
    except (TypeError, ValueError):
        return False


def _first_column_table(node: exp.Expression, aliases: dict[str, str]) -> str | None:
    for col in node.find_all(exp.Column):
        if col.table:
            return aliases.get(col.table, col.table)
    return None


def _has_left_join_to_table(tree: exp.Expression, table: str, aliases: dict[str, str]) -> bool:
    for join in tree.find_all(exp.Join):
        joined = join.this
        if not isinstance(joined, exp.Table):
            continue
        joined_table = aliases.get(joined.alias_or_name, joined.name)
        side = str(join.args.get("side") or "").upper()
        if joined_table == table and side == "LEFT":
            return True
    return False


def _same_resolved_ref(
    left: str | None,
    right: str | None,
    registry: SchemaRegistry,
    signature: SQLSignature,
) -> bool:
    left_resolved = _resolve_ref(left, registry, signature)
    right_resolved = _resolve_ref(right, registry, signature)
    return bool(left_resolved and right_resolved and left_resolved == right_resolved)


def _refs_identify_same_grain(
    left: str | None,
    right: str | None,
    registry: SchemaRegistry,
    signature: SQLSignature,
) -> bool:
    left_resolved = _resolve_ref(left, registry, signature)
    right_resolved = _resolve_ref(right, registry, signature)
    if not left_resolved or not right_resolved:
        return False
    if left_resolved == right_resolved:
        return True
    left_table, left_col = left_resolved
    right_table, right_col = right_resolved
    if left_table in registry.tables:
        fk = registry.fk_to_parent(left_table, right_table)
        if fk and left_col in fk.columns and right_col in fk.ref_columns:
            return True
    if right_table in registry.tables:
        fk = registry.fk_to_parent(right_table, left_table)
        if fk and right_col in fk.columns and left_col in fk.ref_columns:
            return True
    return False


def _resolve_ref(
    ref: str | None,
    registry: SchemaRegistry,
    signature: SQLSignature | None = None,
) -> tuple[str, str] | None:
    if not ref:
        return None
    if "." not in ref:
        if signature is None:
            return None
        matches: list[tuple[str, str]] = []
        for table in signature.tables:
            if table not in registry.tables:
                continue
            column = registry.resolve_column_name(table, ref)
            if column in registry.tables[table].columns:
                matches.append((table, column))
        return matches[0] if len(matches) == 1 else None
    try:
        return registry.table_for_column_ref(ref)
    except KeyError:
        return None
