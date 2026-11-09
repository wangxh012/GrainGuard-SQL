from __future__ import annotations

from dataclasses import dataclass

from .contract import MeasureContract
from .schema_registry import ForeignKey, SchemaRegistry
from .signature_extractor import SQLSignature

SYNTHETIC_FAULT_CHILD = "__synthetic_fault__"
SYNTHETIC_AGGREGATION_SPLIT = "__synthetic_aggregation_split__"


@dataclass(frozen=True)
class TestPlan:
    test_type: str
    measure: MeasureContract
    child_table: str | None = None
    static_status: str | None = None
    pass_class: str | None = None
    diagnosis: str | None = None
    replication_path: tuple[str, ...] = ()
    entity_table: str | None = None


def match_tests(
    measure: MeasureContract,
    signature: SQLSignature,
    registry: SchemaRegistry,
    output_group_ref: str | None = None,
) -> list[TestPlan]:
    plans: list[TestPlan] = []
    requested = set(measure.applicable_tests)

    plans.extend(
        _entity_list_replication_plans(
            measure,
            signature,
            registry,
            output_group_ref,
        )
    )

    if requested.intersection({"CRT", "GRT"}) and measure.semantic_formula == "SUM_ONCE_PER_GRAIN":
        matched_pair = False
        test_type = "GRT" if "GRT" in requested else "CRT"
        for child, uniqueness_basis in _joined_replication_tables(
            measure.source_table,
            signature,
            registry,
        ):
            if _has_agg_on_source(signature, "SUM", measure.source_ref, registry):
                if uniqueness_basis:
                    plans.append(
                        _schema_pass_plan(
                            test_type,
                            measure,
                            child,
                            uniqueness_basis,
                        )
                    )
                    matched_pair = True
                    continue
                plans.append(TestPlan(test_type, measure, child_table=child))
                matched_pair = True
        direct_children = {plan.child_table for plan in plans}
        for child, path in _query_join_replication_paths(
            measure.source_table,
            signature,
            registry,
        ):
            if child in direct_children:
                continue
            plans.append(
                TestPlan(
                    test_type,
                    measure,
                    child_table=child,
                    replication_path=tuple(path),
                )
            )
            matched_pair = True
        if not matched_pair:
            if signature.synthetic_fault and _has_agg_on_source(
                signature, "SUM", measure.source_ref, registry
            ):
                plans.append(TestPlan(test_type, measure, child_table=SYNTHETIC_FAULT_CHILD))
            elif _needs_sum_split_witness(measure, signature, registry):
                plans.append(
                    TestPlan(
                        test_type,
                        measure,
                        child_table=SYNTHETIC_AGGREGATION_SPLIT,
                    )
                )
            else:
                plans.append(
                    _schema_pass_no_witness(
                        test_type,
                        measure,
                        "SQL does not join a lower-grain table that can duplicate the tested grain",
                    )
                )

    if requested.intersection({"CGR", "GRT"}) and measure.semantic_formula == "COUNT_ONCE_PER_GRAIN":
        matched_pair = False
        for child, uniqueness_basis in _joined_replication_tables(
            measure.source_table,
            signature,
            registry,
        ):
            if _has_count_on_grain(signature, measure, registry):
                if uniqueness_basis:
                    plans.append(
                        _schema_pass_plan(
                            "CGR",
                            measure,
                            child,
                            uniqueness_basis,
                        )
                    )
                    matched_pair = True
                    continue
                plans.append(TestPlan("CGR", measure, child_table=child))
                matched_pair = True
        direct_children = {plan.child_table for plan in plans}
        for child, path in _query_join_replication_paths(
            measure.source_table,
            signature,
            registry,
        ):
            if child in direct_children:
                continue
            plans.append(
                TestPlan(
                    "CGR",
                    measure,
                    child_table=child,
                    replication_path=tuple(path),
                )
            )
            matched_pair = True
        if not matched_pair:
            if signature.synthetic_fault and _has_count_on_grain(signature, measure, registry):
                plans.append(TestPlan("CGR", measure, child_table=SYNTHETIC_FAULT_CHILD))
            else:
                plans.append(
                    _schema_pass_no_witness(
                        "CGR",
                        measure,
                        "SQL does not join a lower-grain table that can duplicate the counted grain",
                    )
                )

    if "DET" in requested and measure.semantic_formula == "COUNT_DISTINCT_ENTITY":
        event_tables = _det_replication_tables(signature, measure, registry)
        if signature.synthetic_fault:
            plans.append(TestPlan("DET", measure, child_table=SYNTHETIC_FAULT_CHILD))
            return plans
        if not event_tables or not measure.entity_key:
            plans.append(
                _schema_pass_no_witness(
                    "DET",
                    measure,
                    "no target-aligned duplicating event relation is present in the outer query",
                )
            )
        else:
            entity_table, entity_col = measure.entity_key.split(".", 1)
            for event_table in event_tables:
                entity_fk = registry.fk_to_parent(
                    event_table,
                    entity_table,
                    ref_column=entity_col,
                )
                uniqueness_basis = registry.child_uniqueness_basis_under_filters(
                    event_table,
                    entity_table,
                    signature.filters,
                    entity_fk,
                )
                if uniqueness_basis:
                    plans.append(
                        _schema_pass_plan(
                            "DET",
                            measure,
                            event_table,
                            uniqueness_basis,
                        )
                    )
                else:
                    plans.append(TestPlan("DET", measure, child_table=event_table))

    if "EST" in requested and measure.semantic_formula in {"AVG_ENTITY_TOTAL", "AVG_ONCE_PER_ENTITY"}:
        if _has_agg_on_source(signature, "AVG", measure.source_ref, registry):
            plans.append(TestPlan("EST", measure, child_table=measure.event_table or measure.source_table))
        else:
            event_table = measure.event_table or measure.source_table
            if event_table:
                plans.append(TestPlan("EST", measure, child_table=event_table))
            else:
                plans.append(
                    _schema_pass_no_witness(
                        "EST",
                        measure,
                        "no event table is available for an entity-split witness",
                    )
                )

    if "ART" in requested and measure.semantic_formula == "AVG_ONCE_PER_GRAIN":
        matched_pair = False
        for child, uniqueness_basis in _joined_replication_tables(
            measure.source_table,
            signature,
            registry,
        ):
            if _has_agg_on_source(signature, "AVG", measure.source_ref, registry):
                if uniqueness_basis:
                    plans.append(
                        _schema_pass_plan(
                            "ART",
                            measure,
                            child,
                            uniqueness_basis,
                        )
                    )
                    matched_pair = True
                    continue
                plans.append(TestPlan("ART", measure, child_table=child))
                matched_pair = True
        if not matched_pair:
            if signature.synthetic_fault and _has_agg_on_source(
                signature, "AVG", measure.source_ref, registry
            ):
                plans.append(TestPlan("ART", measure, child_table=SYNTHETIC_FAULT_CHILD))
            else:
                plans.append(
                    _schema_pass_no_witness(
                        "ART",
                        measure,
                        "SQL does not join a lower-grain table that can reweight the averaged grain",
                    )
                )

    if "TST" in requested and measure.semantic_formula == "TOPK_BY_METRIC":
        plans.append(TestPlan("TST", measure))

    return plans


def _entity_list_replication_plans(
    measure: MeasureContract,
    signature: SQLSignature,
    registry: SchemaRegistry,
    output_group_ref: str | None = None,
) -> list[TestPlan]:
    """Match entity-list outputs whose joined detail rows can duplicate them."""

    if (
        signature.measures
        or signature.groups
        or signature.distinct
        # A top-k/detail-row query may legitimately return one entity display
        # value after joining multiple detail rows.  ELT is for unbounded
        # entity-list outputs, not ranking selection.
        or signature.has_order_limit
        # Complex subqueries require a modeled comparison target.  Without
        # one, duplicating an outer detail row can change semantics that the
        # selected contract does not preserve.
        or (
            signature.has_subquery
            and not signature.scalar_aggregate_comparisons
        )
    ):
        return []
    entity_table = None
    if measure.entity_key and "." in measure.entity_key:
        entity_table = measure.entity_key.split(".", 1)[0]
    elif output_group_ref and "." in output_group_ref:
        entity_table = output_group_ref.split(".", 1)[0]
    else:
        entity_table = measure.source_table
    entity_table = registry.resolve_table_name(entity_table)
    projected_tables = {
        table
        for ref in signature.direct_projections
        if (table := _projection_table(ref, signature, registry))
    }
    if entity_table not in projected_tables:
        return []

    plans: list[TestPlan] = []
    for child, uniqueness_basis in _joined_replication_tables(
        entity_table,
        signature,
        registry,
    ):
        if child in projected_tables:
            continue
        if uniqueness_basis:
            continue
        # Scalar-subquery comparisons need to target this child so the stress
        # pair remains relevant.  For ordinary WHERE predicates, the joined
        # child itself is sufficient: duplicating one qualifying detail row
        # preserves the entity set but duplicates the entity-only output.
        if (
            signature.scalar_aggregate_comparisons
            and not any(
                _scalar_comparison_targets_table(
                    item,
                    child,
                    signature,
                    registry,
                )
                for item in signature.scalar_aggregate_comparisons
            )
        ):
            continue
        plans.append(
            TestPlan(
                "ELT",
                measure,
                child_table=child,
                entity_table=entity_table,
            )
        )
    return plans


def _projection_table(
    ref: str,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> str | None:
    if "." in ref:
        return registry.resolve_table_name(ref.split(".", 1)[0])
    matches = [
        table
        for table in (signature.outer_tables or signature.tables)
        if table in registry.tables
        and registry.resolve_column_name(table, ref)
        in registry.tables[table].columns
    ]
    return matches[0] if len(matches) == 1 else None


def _scalar_comparison_targets_table(
    comparison,
    table: str,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> bool:
    source_column = comparison.source.rsplit(".", 1)[-1]
    source_table = _projection_table(comparison.source, signature, registry)
    if source_table != table:
        return False
    if not comparison.aggregate_source:
        return False
    aggregate_column = comparison.aggregate_source.rsplit(".", 1)[-1]
    return (
        registry.resolve_column_name(table, source_column)
        == registry.resolve_column_name(table, aggregate_column)
    )


def _has_agg_on_source(
    signature: SQLSignature,
    function: str,
    source: str | None,
    registry: SchemaRegistry,
) -> bool:
    """Return whether an outer aggregate is bound to the contract source.

    Column-name equality alone is not sufficient.  In particular, an
    unqualified ``points`` selected from ``results`` must not satisfy a
    contract whose source is ``driverStandings.points``.  Resolve an
    unqualified SQL column against the outer relations before comparing it to
    a qualified contract source.
    """
    for sql_measure in signature.measures:
        if sql_measure.function != function:
            continue
        if not source:
            return True
        if _aggregate_source_matches(
            sql_measure.source,
            source,
            signature,
            registry,
        ):
            return True
    return False


def _aggregate_source_matches(
    actual: str | None,
    expected: str,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> bool:
    if not actual:
        return False
    if "." not in expected:
        return _same_ref(actual, expected)
    expected_table, expected_column = expected.split(".", 1)
    expected_table = registry.resolve_table_name(expected_table)
    expected_column = registry.resolve_column_name(
        expected_table,
        expected_column,
    )
    actual_column = actual.rsplit(".", 1)[-1]
    if "." in actual:
        actual_table = registry.resolve_table_name(actual.split(".", 1)[0])
    else:
        actual_table = _resolved_source_table(actual, signature, registry)
    if not actual_table:
        return False
    actual_column = registry.resolve_column_name(actual_table, actual_column)
    return (
        actual_table.casefold() == expected_table.casefold()
        and actual_column.casefold() == expected_column.casefold()
    )


def _schema_pass_plan(
    test_type: str,
    measure: MeasureContract,
    table: str,
    uniqueness_basis: str = "SCHEMA",
) -> TestPlan:
    pass_class = (
        "BENCHMARK_PROFILE_PASS"
        if uniqueness_basis == "BENCHMARK_PROFILE"
        else "SCHEMA_PASS"
    )
    prefix = (
        "benchmark-profile pass"
        if pass_class == "BENCHMARK_PROFILE_PASS"
        else "schema pass"
    )
    return TestPlan(
        test_type,
        measure,
        child_table=table,
        static_status="PASS",
        pass_class=pass_class,
        diagnosis=(
            f"{prefix}: joined table {table} is unique for the tested grain, "
            "so join replication cannot occur"
        ),
    )


def _schema_pass_no_witness(test_type: str, measure: MeasureContract, diagnosis: str) -> TestPlan:
    return TestPlan(
        test_type,
        measure,
        static_status="PASS",
        pass_class="SCHEMA_PASS",
        diagnosis=f"schema pass: {diagnosis}",
    )


def _joined_replication_tables(
    source_table: str,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> list[tuple[str, str | None]]:
    # Replication is an outer-query property. A table used only inside a
    # scalar/IN subquery must not be treated as if it were joined to the outer
    # aggregate. ``outer_tables`` is empty for hand-built signatures in older
    # callers/tests, so retain the previous fallback there.
    relation_tables = signature.outer_tables or signature.tables
    tables: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for child, fk in registry.children_of(source_table):
        if child in seen:
            continue
        if child != source_table and fk.many_allowed and child in relation_tables:
            tables.append(
                (
                    child,
                    registry.child_uniqueness_basis_under_filters(
                        child,
                        source_table,
                        signature.filters,
                        fk,
                    ),
                )
            )
            seen.add(child)

    source_parents = {
        (fk.ref_table, tuple(fk.ref_columns))
        for fk in registry.tables[source_table].foreign_keys
        if fk.many_allowed
    }
    if not source_parents:
        return tables
    for table in relation_tables:
        if table == source_table or table in seen or table not in registry.tables:
            continue
        for fk in registry.tables[table].foreign_keys:
            if fk.many_allowed and (fk.ref_table, tuple(fk.ref_columns)) in source_parents:
                tables.append(
                    (
                        table,
                        registry.sibling_uniqueness_basis_for_source_under_filters(
                            table,
                            source_table,
                            signature.filters,
                            fk,
                        ),
                    )
                )
                seen.add(table)
                break
    return tables


def _query_join_replication_paths(
    source_table: str,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> list[tuple[str, list[str]]]:
    """Find multi-hop outer-query joins that can multiply source-grain rows."""
    relation_tables = signature.outer_tables or signature.tables
    if source_table not in relation_tables:
        return []
    graph: dict[str, list[tuple[str, bool]]] = {
        table: [] for table in relation_tables if table in registry.tables
    }
    for left_ref, right_ref in signature.join_column_pairs:
        if "." not in left_ref or "." not in right_ref:
            continue
        left_table, left_col = left_ref.split(".", 1)
        right_table, right_col = right_ref.split(".", 1)
        left_table = registry.resolve_table_name(left_table)
        right_table = registry.resolve_table_name(right_table)
        if left_table not in graph or right_table not in graph:
            continue
        forward = _join_edge_can_fan_out(
            left_table, left_col, right_table, right_col, signature, registry
        )
        backward = _join_edge_can_fan_out(
            right_table, right_col, left_table, left_col, signature, registry
        )
        if forward is not None:
            graph[left_table].append((right_table, forward))
        if backward is not None:
            graph[right_table].append((left_table, backward))

    queue: list[tuple[str, bool, list[str]]] = [(source_table, False, [source_table])]
    seen: set[tuple[str, bool]] = {(source_table, False)}
    paths: dict[str, list[str]] = {}
    while queue:
        table, has_fanout, path = queue.pop(0)
        for neighbor, edge_fanout in graph.get(table, []):
            if neighbor in path:
                continue
            next_fanout = has_fanout or edge_fanout
            state = (neighbor, next_fanout)
            if state in seen:
                continue
            seen.add(state)
            next_path = [*path, neighbor]
            if next_fanout and len(next_path) > 2:
                paths.setdefault(neighbor, next_path)
            queue.append((neighbor, next_fanout, next_path))
    return sorted(paths.items())


def _join_edge_can_fan_out(
    source_table: str,
    source_col: str,
    target_table: str,
    target_col: str,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> bool | None:
    source_col = registry.resolve_column_name(source_table, source_col)
    target_col = registry.resolve_column_name(target_table, target_col)
    for fk in registry.fks_to_parent(target_table, source_table):
        if target_col in fk.columns and source_col in fk.ref_columns:
            return not bool(
                registry.child_uniqueness_basis_under_filters(
                    target_table, source_table, signature.filters, fk
                )
            )
    for fk in registry.fks_to_parent(source_table, target_table):
        if source_col in fk.columns and target_col in fk.ref_columns:
            return False
    for source_fk in registry.tables[source_table].foreign_keys:
        if source_col not in source_fk.columns:
            continue
        for target_fk in registry.tables[target_table].foreign_keys:
            if target_col not in target_fk.columns:
                continue
            if (
                source_fk.ref_table == target_fk.ref_table
                and source_fk.ref_columns == target_fk.ref_columns
            ):
                return not bool(
                    registry.sibling_uniqueness_basis_for_source_under_filters(
                        target_table, source_table, signature.filters, target_fk
                    )
                )
    return None


def _needs_sum_split_witness(
    measure: MeasureContract,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> bool:
    """Detect missing SUMs and SUM/COUNT unit mismatches."""
    has_sum = _has_agg_on_source(
        signature, "SUM", measure.source_ref, registry
    )
    if not has_sum:
        if any(
            any(
                _aggregate_source_matches(group, grain, signature, registry)
                for grain in measure.base_grain
            )
            for group in signature.groups
        ):
            return False
        if any(
            item.operator in {"=", "EQ"}
            and _resolved_source_table(item.source, signature, registry)
            == measure.source_table
            for item in signature.filters
        ):
            return False
        return any(
            _aggregate_source_matches(
                ref, measure.source_ref or "", signature, registry
            )
            for ref in signature.direct_projections
        )
    return any(
        item.function == "COUNT"
        and not _aggregate_source_matches(
            item.source, measure.source_ref or "", signature, registry
        )
        for item in signature.measures
    )


def _has_count_without_distinct(signature: SQLSignature, measure: MeasureContract, registry: SchemaRegistry) -> bool:
    return bool(_det_row_count_replication_tables(signature, measure, registry))


def _det_replication_tables(
    signature: SQLSignature,
    measure: MeasureContract,
    registry: SchemaRegistry,
) -> list[str]:
    event_tables = _det_row_count_replication_tables(signature, measure, registry)
    if event_tables:
        return event_tables

    # A non-aggregate entity list can still expose bag duplication, but only
    # when the outer projection stays at entity grain. Detail projections make
    # repeated entity values legitimate and are not target-aligned DET evidence.
    if (
        measure.entity_key
        and not signature.measures
        and not signature.groups
        and not signature.distinct
    ):
        entity_table, _ = measure.entity_key.split(".", 1)
        projection_tables = {
            ref.split(".", 1)[0]
            for ref in signature.projections
            if "." in ref
        }
        if projection_tables and projection_tables <= {entity_table}:
            return [
                child
                for child, _ in _entity_replication_children(
                    signature,
                    measure,
                    registry,
                )
            ]
    return []


def _det_row_count_replication_tables(
    signature: SQLSignature,
    measure: MeasureContract,
    registry: SchemaRegistry,
) -> list[str]:
    if not measure.entity_key:
        return []
    children = _entity_replication_children(signature, measure, registry)
    if not children:
        return []
    child_names = {child for child, _ in children}
    entity_table, _ = measure.entity_key.split(".", 1)
    matched: list[str] = []
    for sql_measure in signature.measures:
        if not sql_measure.row_counting:
            continue
        if sql_measure.source is None:
            if not sql_measure.distinct:
                matched.extend(child_names)
            continue
        source_column = sql_measure.source.rsplit(".", 1)[-1]
        entity_column = measure.entity_key.rsplit(".", 1)[-1]
        source_table = _resolved_source_table(
            sql_measure.source,
            signature,
            registry,
        )
        if source_table == entity_table:
            if not sql_measure.distinct:
                matched.extend(child_names)
        elif source_table in child_names:
            child_fk = next(
                fk for child, fk in children if child == source_table
            )
            if not (
                sql_measure.distinct and source_column in child_fk.columns
            ):
                matched.append(source_table)
        elif "." not in sql_measure.source:
            # An unqualified identifier can be ambiguous in the original SQL.
            # If it names the contract entity key, validation's qualification
            # retry treats it as the entity column; otherwise bind it only when
            # exactly one joined detail relation contains that column.
            if source_column.casefold() == entity_column.casefold():
                if not sql_measure.distinct:
                    matched.extend(child_names)
                continue
            column_children = [
                child
                for child in child_names
                if registry.resolve_column_name(child, source_column)
                in registry.tables[child].columns
            ]
            if len(column_children) == 1:
                matched.extend(column_children)
    return list(dict.fromkeys(matched))


def _entity_replication_children(
    signature: SQLSignature,
    measure: MeasureContract,
    registry: SchemaRegistry,
) -> list[tuple[str, ForeignKey]]:
    if not measure.entity_key:
        return []
    entity_table, entity_col = measure.entity_key.split(".", 1)
    entity_table = registry.resolve_table_name(entity_table)
    entity_col = registry.resolve_column_name(entity_table, entity_col)
    entity_identifiers = registry.entity_identifier_columns(entity_table)
    relation_tables = signature.outer_tables or signature.tables
    children: list[tuple[str, ForeignKey]] = []
    for child, fk in registry.children_of(entity_table):
        if child not in relation_tables or not fk.many_allowed:
            continue
        referenced_columns = {
            registry.resolve_column_name(entity_table, column)
            for column in fk.ref_columns
        }
        if entity_col not in referenced_columns and not (
            entity_col in entity_identifiers
            and referenced_columns
            and referenced_columns <= entity_identifiers
        ):
            continue
        children.append((child, fk))
    return children


def _count_star_over_event_entity_join(signature: SQLSignature, measure: MeasureContract) -> bool:
    if not measure.entity_key or not measure.event_table:
        return False
    entity_table, _ = measure.entity_key.split(".", 1)
    relation_tables = signature.outer_tables or signature.tables
    return measure.event_table in relation_tables and entity_table in relation_tables


def _counts_event_fk_to_entity(source: str | None, measure: MeasureContract, registry: SchemaRegistry) -> bool:
    if not source or not measure.entity_key or not measure.event_table:
        return False
    source_table, source_col = source.split(".", 1) if "." in source else ("", source)
    entity_table, entity_col = measure.entity_key.split(".", 1)
    if source_table and source_table != measure.event_table:
        return False
    fk = registry.fk_to_parent(
        measure.event_table,
        entity_table,
        ref_column=entity_col,
    )
    return bool(fk and source_col in fk.columns)


def _has_count_on_grain(signature: SQLSignature, measure: MeasureContract, registry: SchemaRegistry) -> bool:
    accepted_sources = {measure.source_ref, *measure.base_grain}
    measure_table = measure.source_table
    for sql_measure in signature.measures:
        if sql_measure.function != "COUNT" or sql_measure.distinct:
            continue
        if sql_measure.source is None:
            return True
        if any(_same_ref(sql_measure.source, source) for source in accepted_sources if source):
            return True
        if any(
            registry.columns_identify_same_entity(sql_measure.source, source)
            for source in accepted_sources
            if source
        ):
            return True
        if _counts_child_fk_to_grain(sql_measure.source, measure, registry):
            return True
        source_table = _resolved_source_table(sql_measure.source, signature, registry)
        if measure_table in signature.tables and source_table and source_table != measure_table:
            return True
    return False


def _resolved_source_table(
    source: str,
    signature: SQLSignature,
    registry: SchemaRegistry,
) -> str:
    if "." in source:
        return source.split(".", 1)[0]
    matches = [
        table
        for table in (signature.outer_tables or signature.tables)
        if table in registry.tables and source in registry.tables[table].columns
    ]
    return matches[0] if len(matches) == 1 else ""


def _counts_child_fk_to_grain(source: str | None, measure: MeasureContract, registry: SchemaRegistry) -> bool:
    if not source:
        return False
    source_table, source_col = source.split(".", 1) if "." in source else ("", source)
    if source_table == measure.source_table:
        return False
    fk = registry.fk_to_parent(source_table, measure.source_table) if source_table in registry.tables else None
    return bool(fk and source_col in fk.columns)


def _same_ref(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    if left.lower() == right.lower():
        return True
    left_col = left.rsplit(".", 1)[-1].lower()
    right_col = right.rsplit(".", 1)[-1].lower()
    return "." not in left and left_col == right_col
