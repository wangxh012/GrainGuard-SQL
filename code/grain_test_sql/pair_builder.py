from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contract import MetricContract, MeasureContract
from .schema_registry import ForeignKey, SchemaRegistry, default_value
from .signature_extractor import FilterSignature, ScalarAggregateComparison
from .template_matcher import (
    SYNTHETIC_AGGREGATION_SPLIT,
    SYNTHETIC_FAULT_CHILD,
    TestPlan,
)


Rows = dict[str, list[dict[str, Any]]]


@dataclass
class TestPair:
    test_type: str
    measure: MeasureContract
    d0: Rows
    d1: Rows
    group_value: Any = "East"
    meta: dict[str, Any] = field(default_factory=dict)


class PairBuilder:
    def __init__(
        self,
        registry: SchemaRegistry,
        contract: MetricContract,
        filters: list[FilterSignature] | None = None,
        conditional_filters: list[FilterSignature] | None = None,
        required_tables: set[str] | None = None,
        join_column_pairs: list[tuple[str, str]] | None = None,
        group_columns: list[str] | None = None,
        scalar_aggregate_comparisons: list[ScalarAggregateComparison] | None = None,
        projection_columns: list[str] | None = None,
        ranking_selection: bool = False,
        ranking_count_sources: list[str | None] | None = None,
        enable_cgr_group_ranking_contrast: bool = True,
    ):
        self.registry = registry
        self.contract = contract
        self.filters = filters or []
        self.conditional_filters = conditional_filters or []
        self.required_tables = required_tables or set()
        self.join_column_pairs = join_column_pairs or []
        self.group_columns = group_columns or []
        self.scalar_aggregate_comparisons = scalar_aggregate_comparisons or []
        self.projection_columns = projection_columns or []
        self.ranking_selection = ranking_selection
        self.ranking_count_sources = ranking_count_sources or []
        self.enable_cgr_group_ranking_contrast = enable_cgr_group_ranking_contrast

    def build(self, plan: TestPlan) -> TestPair:
        if plan.test_type in {"CRT", "GRT", "CGR"}:
            return self._build_grt(plan)
        if plan.test_type == "DET":
            return self._build_det(plan)
        if plan.test_type == "EST":
            return self._build_est(plan)
        if plan.test_type == "ART":
            return self._build_art(plan)
        if plan.test_type == "TST":
            return self._build_tst(plan)
        if plan.test_type == "ELT":
            return self._build_entity_list_replication(plan)
        raise ValueError(f"Unsupported test type: {plan.test_type}")

    def _build_entity_list_replication(self, plan: TestPlan) -> TestPair:
        """Reassign detail rows while preserving their values and entity set.

        A one-entity D0 followed by a duplicated detail row in D1 only
        preserves contracts whose value is already duplicate-insensitive.  In
        particular, it cannot validate an entity-list defect beside a COUNT,
        SUM, or AVG target over the detail table.  Use two output entities and
        move one of three existing detail rows between them instead.  The
        detail-row multiset and both entity domains stay fixed, while an
        unbounded entity-only projection exposes the changed multiplicities.
        """

        measure = plan.measure
        child = plan.child_table
        if not child:
            raise ValueError("ELT requires a child table")
        entity_table = plan.entity_table or (
            measure.entity_key.split(".", 1)[0]
            if measure.entity_key and "." in measure.entity_key
            else measure.source_table
        )
        entity_table = self.registry.resolve_table_name(entity_table)

        d0 = self._empty_rows()
        entity_a = self._base_row(entity_table, index=1)
        entity_b = self._base_row(entity_table, index=2)
        self._differentiate_entity_projection(entity_table, entity_b, index=2)
        for entity in (entity_a, entity_b):
            self._ensure_parent_chain(d0, entity_table, entity)
            self._append_once(d0, entity_table, entity)

        detail_a = self._base_row(child, index=1)
        self._link_child_to_parent(child, detail_a, entity_table, entity_a)
        self._ensure_parent_chain(d0, child, detail_a)
        self._append_once(d0, child, detail_a)
        for index in (2, 3):
            detail_b = self._variant_row(child, detail_a, index=index)
            self._link_child_to_parent(child, detail_b, entity_table, entity_b)
            self._copy_group_values(child, detail_b, detail_a)
            self._ensure_parent_chain(d0, child, detail_b)
            self._append_once(d0, child, detail_b)

        d1 = self._clone(d0)
        reassigned = d1[child][-1]
        self._link_child_to_parent(child, reassigned, entity_table, entity_a)

        pair = TestPair(
            "ELT",
            measure,
            d0,
            d1,
            meta={
                "child_table": child,
                "entity_table": entity_table,
                "preserve_output_group_domain": True,
            },
        )
        self._apply_scalar_aggregate_entity_list_stress(
            pair,
            entity_table,
            child,
        )
        self._ensure_filter_context(pair.d0, child)
        self._ensure_filter_context(pair.d1, child)
        self._ensure_required_tables(pair.d0, child)
        self._ensure_required_tables(pair.d1, child)
        return self._with_filters(pair)

    def _differentiate_entity_projection(
        self,
        entity_table: str,
        row: dict[str, Any],
        *,
        index: int,
    ) -> None:
        """Give two entity rows different projected display values."""

        for ref in self.projection_columns:
            if "." in ref:
                table, raw_column = ref.split(".", 1)
                if self.registry.resolve_table_name(table) != entity_table:
                    continue
                column = self.registry.resolve_column_name(entity_table, raw_column)
            else:
                matches = [
                    table
                    for table in self.required_tables
                    if table in self.registry.tables
                    and self.registry.resolve_column_name(table, ref)
                    in self.registry.tables[table].columns
                ]
                if len(matches) != 1 or matches[0] != entity_table:
                    continue
                column = self.registry.resolve_column_name(entity_table, ref)
            if column not in row:
                continue
            row[column] = default_value(
                entity_table,
                column,
                self.registry.tables[entity_table].columns[column],
                index=index,
            )

    def _apply_scalar_aggregate_entity_list_stress(
        self,
        pair: TestPair,
        entity_table: str,
        child: str,
    ) -> None:
        """Make a scalar-average predicate non-vacuous in both databases."""

        comparison = next(
            (
                item
                for item in self.scalar_aggregate_comparisons
                if item.aggregate_function == "AVG"
                and item.operator in {"<", "<=", ">", ">="}
                and self._comparison_targets_child_column(item, child)
            ),
            None,
        )
        if comparison is None:
            return
        column = comparison.source.rsplit(".", 1)[-1]
        target_value, contrast_value = (
            (10, 30)
            if comparison.operator in {"<", "<="}
            else (30, 10)
        )
        for rows in (pair.d0, pair.d1):
            for row in rows.get(child) or []:
                row[column] = target_value

            contrast_index = len(rows.get(child) or []) + 1
            contrast_entity = self._base_row(entity_table, index=contrast_index)
            self._ensure_parent_chain(rows, entity_table, contrast_entity)
            self._append_once(rows, entity_table, contrast_entity)
            contrast_detail = self._base_row(child, index=contrast_index)
            self._link_child_to_parent(
                child,
                contrast_detail,
                entity_table,
                contrast_entity,
            )
            contrast_detail[column] = contrast_value
            self._ensure_parent_chain(rows, child, contrast_detail)
            self._append_once(rows, child, contrast_detail)

    def _comparison_targets_child_column(
        self,
        comparison: ScalarAggregateComparison,
        child: str,
    ) -> bool:
        source_column = comparison.source.rsplit(".", 1)[-1]
        if "." in comparison.source:
            source_table = self.registry.resolve_table_name(
                comparison.source.split(".", 1)[0]
            )
        else:
            containing = [
                table
                for table in self.required_tables
                if table in self.registry.tables
                and self.registry.resolve_column_name(table, source_column)
                in self.registry.tables[table].columns
            ]
            source_table = containing[0] if len(containing) == 1 else None
        if source_table != child:
            return False
        aggregate_source = comparison.aggregate_source
        if not aggregate_source:
            return False
        aggregate_column = aggregate_source.rsplit(".", 1)[-1]
        return (
            self.registry.resolve_column_name(child, source_column)
            == self.registry.resolve_column_name(child, aggregate_column)
        )

    def _empty_rows(self) -> Rows:
        return {table: [] for table in self.registry.tables}

    def _base_row(self, table: str, index: int = 1) -> dict[str, Any]:
        schema = self.registry.tables[table]
        row = {
            col: default_value(table, col, typ, index=index)
            for col, typ in schema.columns.items()
        }
        if table == self.contract.output_group.table and "text" in schema.columns[self.contract.output_group.column].lower():
            row[self.contract.output_group.column] = "East"
        return row

    def _ensure_parent_chain(self, rows: Rows, table: str, row: dict[str, Any], seen: set[str] | None = None) -> None:
        seen = seen or set()
        if table in seen:
            return
        seen.add(table)
        for fk in self.registry.tables[table].foreign_keys:
            child_key = tuple(row.get(column) for column in fk.columns)
            if all(value is not None for value in child_key):
                existing_parent = next(
                    (
                        parent
                        for parent in rows.get(fk.ref_table, [])
                        if tuple(parent.get(column) for column in fk.ref_columns)
                        == child_key
                    ),
                    None,
                )
                if existing_parent is not None:
                    self._ensure_parent_chain(
                        rows,
                        fk.ref_table,
                        existing_parent,
                        seen,
                    )
                    continue
            parent_index = len(rows.get(fk.ref_table) or []) + 1
            parent_row = self._base_row(fk.ref_table, index=parent_index)
            for child_col, parent_col in zip(fk.columns, fk.ref_columns):
                if row.get(child_col) is not None:
                    parent_row[parent_col] = row[child_col]
                else:
                    row[child_col] = parent_row[parent_col]
            self._ensure_parent_chain(rows, fk.ref_table, parent_row, seen)
            self._append_once(rows, fk.ref_table, parent_row)

    def _append_once(self, rows: Rows, table: str, row: dict[str, Any]) -> None:
        pk = self.registry.tables[table].primary_key
        if pk and any(all(existing[c] == row[c] for c in pk) for existing in rows[table]):
            return
        rows[table].append(row)

    def _append_duplicate_allowed(self, rows: Rows, table: str, row: dict[str, Any]) -> None:
        rows[table].append(row)

    def _clone(self, rows: Rows) -> Rows:
        return {table: [dict(row) for row in table_rows] for table, table_rows in rows.items()}

    def _build_grt(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        child = plan.child_table
        if child == SYNTHETIC_AGGREGATION_SPLIT:
            return self._build_synthetic_grt_count_fault(plan)
        if plan.replication_path:
            return self._build_path_replication(plan)
        if child == SYNTHETIC_FAULT_CHILD:
            return self._build_synthetic_grain_fault(plan)
        if not child:
            raise ValueError("CRT requires child_table")
        d0 = self._empty_rows()
        parent_row = self._base_row(measure.source_table)
        parent_row[measure.source_column] = 100
        self._ensure_parent_chain(d0, measure.source_table, parent_row)
        self._append_once(d0, measure.source_table, parent_row)

        child_row = self._base_row(child)
        self._link_child_to_parent(child, child_row, measure.source_table, parent_row)
        self._ensure_parent_chain(d0, child, child_row)
        self._append_once(d0, child, child_row)

        d1 = self._clone(d0)
        second_child = self._base_row(child, index=2)
        self._link_child_to_parent(child, second_child, measure.source_table, parent_row)
        self._copy_group_values(child, second_child, child_row)
        self._ensure_parent_chain(d1, child, second_child)
        self._append_duplicate_allowed(d1, child, second_child)
        self._ensure_filter_context(d0, measure.source_table)
        self._ensure_filter_context(d1, measure.source_table)
        self._ensure_required_tables(d0, measure.source_table)
        self._ensure_required_tables(d1, measure.source_table)
        pair = self._with_filters(
            TestPair(plan.test_type, measure, d0, d1, meta={"child_table": child})
        )
        self._apply_conditional_aggregate_stress(pair)
        self._add_cgr_group_ranking_contrast(pair)
        return pair

    def _add_cgr_group_ranking_contrast(self, pair: TestPair) -> None:
        """Expose join fan-out hidden behind a grouped COUNT ranking.

        Replicating a joined child changes the count of a single group, but a
        one-group database cannot change the winner selected by ORDER BY and
        LIMIT.  Add a stable two-row contrast group whose correct count lies
        between the target group's D0 and D1 row-weighted counts.
        """

        measure = pair.measure
        child = pair.meta.get("child_table")
        if (
            pair.test_type != "CGR"
            or not self.enable_cgr_group_ranking_contrast
            or not self.ranking_selection
            or not child
            or child not in self.registry.tables
            or not measure.source_column
            or not self._ranking_counts_measure_source(measure)
            or not pair.d0.get(measure.source_table)
            or not pair.d0.get(child)
            or not pair.d1.get(child)
        ):
            return

        group_ref = next(
            (
                ref
                for ref in self.group_columns
                if "." in ref
                and self.registry.resolve_table_name(ref.split(".", 1)[0])
                == measure.source_table
            ),
            None,
        )
        if group_ref is None:
            return
        group_column = self.registry.resolve_column_name(
            measure.source_table,
            group_ref.split(".", 1)[1],
        )
        if any(
            filter_.operator == "="
            and filter_.source.casefold() == group_ref.casefold()
            for filter_ in self.filters
        ):
            return

        original_source = pair.d0[measure.source_table][0]
        original_child = pair.d0[child][0]

        # Make the target group's faulty D1 count three, so it crosses the
        # contrast group's stable count of two rather than merely tying it.
        third_child = self._variant_row(
            child,
            pair.d1[child][0],
            index=len(pair.d1[child]) + 1,
        )
        self._link_child_to_parent(
            child,
            third_child,
            measure.source_table,
            original_source,
        )
        self._copy_group_values(child, third_child, original_child)
        self._append_duplicate_allowed(pair.d1, child, third_child)

        contrast = self._variant_row(
            measure.source_table,
            original_source,
            index=3,
        )
        contrast[group_column] = self._contrast_value(
            original_source.get(group_column)
        )
        for rows in (pair.d0, pair.d1):
            self._ensure_parent_chain(
                rows,
                measure.source_table,
                dict(contrast),
            )
            self._append_once(
                rows,
                measure.source_table,
                dict(contrast),
            )

        for offset in (5, 6):
            contrast_child = self._variant_row(
                child,
                original_child,
                index=offset,
            )
            self._link_child_to_parent(
                child,
                contrast_child,
                measure.source_table,
                contrast,
            )
            self._set_contrast_child_values(child, contrast_child)
            for rows in (pair.d0, pair.d1):
                self._ensure_parent_chain(rows, child, dict(contrast_child))
                self._append_once(rows, child, dict(contrast_child))

        self._with_filters(pair)

    def _ranking_counts_measure_source(self, measure: MeasureContract) -> bool:
        expected = measure.source_ref.casefold()
        return any(
            source and source.casefold() == expected
            for source in self.ranking_count_sources
        )

    def _set_contrast_child_values(
        self,
        table: str,
        row: dict[str, Any],
    ) -> None:
        """Give a ranking contrast group a distinguishable projected value."""

        protected = {
            column
            for columns in [
                self.registry.tables[table].primary_key,
                *self.registry.tables[table].unique,
                *(fk.columns for fk in self.registry.tables[table].foreign_keys),
            ]
            for column in columns
        }
        protected.update(
            filter_.source.split(".", 1)[1]
            for filter_ in self.filters
            if "." in filter_.source
            and self.registry.resolve_table_name(
                filter_.source.split(".", 1)[0]
            )
            == table
        )
        for column, type_name in self.registry.tables[table].columns.items():
            if column in protected or "text" not in type_name.casefold():
                continue
            row[column] = self._contrast_value(row.get(column))

    @staticmethod
    def _contrast_value(value: Any) -> Any:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value + 100
        return f"{value}_contrast"

    def _build_path_replication(self, plan: TestPlan) -> TestPair:
        """Duplicate the terminal row of a declared multi-hop query join."""
        measure = plan.measure
        child = plan.child_table
        if not child:
            raise ValueError("path replication requires a terminal table")
        d0 = self._empty_rows()
        source_row = self._base_row(measure.source_table)
        if measure.source_column:
            source_row[measure.source_column] = 100
        self._ensure_parent_chain(d0, measure.source_table, source_row)
        self._append_once(d0, measure.source_table, source_row)
        self._ensure_filter_context(d0, measure.source_table)
        self._ensure_required_tables(d0, measure.source_table)
        pair = self._with_filters(
            TestPair(
                plan.test_type,
                measure,
                d0,
                self._clone(d0),
                meta={
                    "child_table": child,
                    "replication_path": list(plan.replication_path),
                },
            )
        )
        if not pair.d0.get(child):
            raise ValueError(
                f"could not materialize {child} on join path "
                f"{' -> '.join(plan.replication_path)}"
            )
        pair.d1 = self._clone(pair.d0)
        original = pair.d1[child][0]
        duplicate = self._variant_row(
            child, original, index=len(pair.d1[child]) + 1
        )
        self._append_duplicate_allowed(pair.d1, child, duplicate)
        pair = self._with_filters(pair)
        self._apply_conditional_aggregate_stress(pair)
        return pair

    def _build_det(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        event_table = plan.child_table or measure.event_table
        if event_table == SYNTHETIC_FAULT_CHILD:
            return self._build_synthetic_det_fault(plan)
        entity_table, entity_col = measure.entity_key.split(".", 1)
        fk = (
            self.registry.fk_to_parent(
                event_table,
                entity_table,
                ref_column=entity_col,
            )
            if event_table
            else None
        )
        event_entity_cols = list(fk.columns) if fk else ([entity_col] if event_table and entity_col in self.registry.tables[event_table].columns else [])
        if not event_table:
            raise ValueError("DET requires event_table")
        d0 = self._empty_rows()
        entity_row = self._base_row(entity_table)
        self._ensure_parent_chain(d0, entity_table, entity_row)
        self._append_once(d0, entity_table, entity_row)
        event_row = self._base_row(event_table)
        self._link_child_to_parent(event_table, event_row, entity_table, entity_row)
        if not self.registry.fk_to_parent(event_table, entity_table) and entity_col in event_row:
            event_row[entity_col] = entity_row[entity_col]
        self._append_once(d0, event_table, event_row)

        d1 = self._clone(d0)
        if not self.registry.columns_are_unique(entity_table, [entity_col]):
            entity_row2 = self._base_row(entity_table, index=2)
            entity_row2[entity_col] = entity_row[entity_col]
            for pk_col in self.registry.tables[entity_table].primary_key:
                if pk_col != entity_col:
                    entity_row2[pk_col] = default_value(
                        entity_table,
                        pk_col,
                        self.registry.tables[entity_table].columns[pk_col],
                        index=2,
                    )
            self._ensure_parent_chain(d1, entity_table, entity_row2)
            self._append_once(d1, entity_table, entity_row2)
            event_row2 = self._variant_row(event_table, event_row, index=2)
            self._link_child_to_parent(event_table, event_row2, entity_table, entity_row2)
        else:
            event_row2 = self._variant_row(event_table, event_row, index=2)
            self._link_child_to_parent(event_table, event_row2, entity_table, entity_row)
            if not self.registry.fk_to_parent(event_table, entity_table) and entity_col in event_row2:
                event_row2[entity_col] = entity_row[entity_col]
        self._copy_group_values(event_table, event_row2, event_row)
        self._append_duplicate_allowed(d1, event_table, event_row2)
        self._ensure_filter_context(d0, event_table)
        self._ensure_filter_context(d1, event_table)
        self._ensure_required_tables(d0, event_table)
        self._ensure_required_tables(d1, event_table)
        pair = self._with_filters(TestPair("DET", measure, d0, d1, meta={"event_table": event_table, "event_entity_cols": event_entity_cols}))
        self._apply_conditional_aggregate_stress(pair)
        self._add_det_group_ranking_contrast(pair)
        return pair

    def _add_det_group_ranking_contrast(self, pair: TestPair) -> None:
        """Expose row-count ranking faults while preserving entity counts.

        A one-group DET pair changes a hidden ``COUNT`` but cannot change a
        projected winner when the count appears only in ``ORDER BY``.  Add a
        stable two-entity contrast group and a second duplicate event for the
        original entity.  The distinct-entity population of every group stays
        fixed, while a row-counting candidate flips its selected group.
        """

        if (
            not self.ranking_selection
            or not self.group_columns
            or not self._ranking_counts_contract_entity(pair)
        ):
            return
        measure = pair.measure
        event_table = pair.meta.get("event_table")
        if (
            not measure.entity_key
            or not event_table
            or event_table not in self.registry.tables
        ):
            return
        entity_table, entity_col = measure.entity_key.split(".", 1)
        entity_table = self.registry.resolve_table_name(entity_table)
        entity_col = self.registry.resolve_column_name(entity_table, entity_col)
        if not pair.d0.get(entity_table) or not pair.d1.get(event_table):
            return

        original_entity = pair.d0[entity_table][0]
        original_event = pair.d1[event_table][0]
        # D1 must cross the contrast group's count rather than merely tie it.
        second_duplicate = self._variant_row(
            event_table,
            original_event,
            index=len(pair.d1[event_table]) + 1,
        )
        self._link_child_to_parent(
            event_table,
            second_duplicate,
            entity_table,
            original_entity,
        )
        self._copy_group_values(event_table, second_duplicate, original_event)
        self._append_duplicate_allowed(pair.d1, event_table, second_duplicate)

        for index in (3, 4):
            contrast_entity = self._variant_row(
                entity_table,
                original_entity,
                index=index,
            )
            contrast_entity[entity_col] = default_value(
                entity_table,
                entity_col,
                self.registry.tables[entity_table].columns[entity_col],
                index=index,
            )
            self._set_contrast_group_values(
                entity_table,
                contrast_entity,
                original_entity,
            )
            contrast_event = self._base_row(event_table, index=index + 2)
            self._link_child_to_parent(
                event_table,
                contrast_event,
                entity_table,
                contrast_entity,
            )
            for rows in (pair.d0, pair.d1):
                self._ensure_parent_chain(
                    rows,
                    entity_table,
                    dict(contrast_entity),
                )
                self._append_once(
                    rows,
                    entity_table,
                    dict(contrast_entity),
                )
                self._ensure_parent_chain(
                    rows,
                    event_table,
                    dict(contrast_event),
                )
                self._append_once(
                    rows,
                    event_table,
                    dict(contrast_event),
                )
        self._with_filters(pair)

    def _ranking_counts_contract_entity(self, pair: TestPair) -> bool:
        """Reject event-count rankings, whose multiplicity is intentional."""

        if not pair.measure.entity_key:
            return False
        entity_table, entity_col = pair.measure.entity_key.split(".", 1)
        entity_table = self.registry.resolve_table_name(entity_table)
        entity_col = self.registry.resolve_column_name(
            entity_table,
            entity_col,
        )
        identifiers = self.registry.entity_identifier_columns(entity_table)
        for source in self.ranking_count_sources:
            if not source:
                continue
            source_column = source.rsplit(".", 1)[-1]
            source_column = self.registry.resolve_column_name(
                entity_table,
                source_column,
            )
            if "." in source:
                source_table = self.registry.resolve_table_name(
                    source.split(".", 1)[0]
                )
                if source_table != entity_table:
                    continue
            elif (
                source_column != entity_col
                and source_column not in self.registry.tables[entity_table].columns
            ):
                continue
            if (
                source_column == entity_col
                or source_column in identifiers
                or source_column in self.registry.tables[entity_table].columns
            ):
                return True
        return False

    def _set_contrast_group_values(
        self,
        table: str,
        target: dict[str, Any],
        original: dict[str, Any],
    ) -> None:
        for ref in self.group_columns:
            if "." in ref:
                ref_table, raw_column = ref.split(".", 1)
                if self.registry.resolve_table_name(ref_table) != table:
                    continue
                column = self.registry.resolve_column_name(table, raw_column)
            else:
                containing = [
                    relation
                    for relation in self.required_tables
                    if relation in self.registry.tables
                    and self.registry.resolve_column_name(relation, ref)
                    in self.registry.tables[relation].columns
                ]
                if len(containing) != 1 or containing[0] != table:
                    continue
                column = self.registry.resolve_column_name(table, ref)
            if column not in target:
                continue
            value = original.get(column)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                target[column] = value + 100
            else:
                target[column] = f"{value}_contrast"

    def _build_synthetic_det_fault(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        entity_table, entity_col = measure.entity_key.split(".", 1)
        fault_col = self._synthetic_entity_fault_column(entity_table, entity_col)
        d0 = self._empty_rows()
        entity_row = self._base_row(entity_table)
        entity_row[fault_col] = self._synthetic_non_trigger_value(entity_table, fault_col)
        self._ensure_parent_chain(d0, entity_table, entity_row)
        self._append_once(d0, entity_table, entity_row)

        d1 = self._clone(d0)
        d1[entity_table][0][fault_col] = self._synthetic_trigger_value(entity_table, fault_col)
        return self._with_filters(
            TestPair(
                "DET",
                measure,
                d0,
                d1,
                meta={"event_table": SYNTHETIC_FAULT_CHILD, "synthetic_fault_column": fault_col},
            )
        )

    def _build_est(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        entity_table, entity_col = (
            measure.entity_key.split(".", 1)
            if measure.entity_key
            else (self.contract.output_group.table, "")
        )
        event_table = measure.event_table or measure.source_table
        fk = self.registry.fk_to_parent(
            event_table,
            entity_table,
            ref_column=entity_col or None,
        )
        event_entity_cols = list(fk.columns) if fk else []
        d0 = self._empty_rows()
        entity_row = self._base_row(entity_table)
        self._ensure_parent_chain(d0, entity_table, entity_row)
        self._append_once(d0, entity_table, entity_row)

        event_row = self._base_row(event_table)
        self._link_child_to_parent(event_table, event_row, entity_table, entity_row)
        event_row[measure.source_column] = 100
        self._append_once(d0, event_table, event_row)

        d1 = self._clone(d0)
        d1[event_table] = [row for row in d1[event_table] if row != event_row]
        event_row1 = self._base_row(event_table, index=1)
        event_row2 = self._base_row(event_table, index=2)
        self._link_child_to_parent(event_table, event_row1, entity_table, entity_row)
        self._link_child_to_parent(event_table, event_row2, entity_table, entity_row)
        event_row1[measure.source_column] = 40
        event_row2[measure.source_column] = 60
        self._append_once(d1, event_table, event_row1)
        self._append_once(d1, event_table, event_row2)
        return self._with_filters(TestPair("EST", measure, d0, d1, meta={"event_table": event_table, "event_entity_cols": event_entity_cols}))

    def _build_art(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        child = plan.child_table
        if child == SYNTHETIC_FAULT_CHILD:
            return self._build_synthetic_art_fault(plan)
        if not child:
            raise ValueError("ART requires child_table")
        d0 = self._empty_rows()

        source_row1 = self._base_row(measure.source_table, index=1)
        source_row2 = self._base_row(measure.source_table, index=2)
        source_row1[measure.source_column] = 100
        source_row2[measure.source_column] = 200
        self._ensure_parent_chain(d0, measure.source_table, source_row1)
        self._ensure_parent_chain(d0, measure.source_table, source_row2)
        self._append_once(d0, measure.source_table, source_row1)
        self._append_once(d0, measure.source_table, source_row2)

        child_row1 = self._base_row(child, index=1)
        child_row2 = self._base_row(child, index=2)
        self._link_child_to_parent(child, child_row1, measure.source_table, source_row1)
        self._link_child_to_parent(child, child_row2, measure.source_table, source_row2)
        self._copy_query_join_context(
            child,
            child_row2,
            child_row1,
            exclude_parent=measure.source_table,
        )
        self._ensure_parent_chain(d0, child, child_row1)
        self._ensure_parent_chain(d0, child, child_row2)
        self._append_once(d0, child, child_row1)
        self._append_once(d0, child, child_row2)

        d1 = self._clone(d0)
        extra_child = self._base_row(child, index=3)
        self._link_child_to_parent(child, extra_child, measure.source_table, source_row2)
        self._copy_query_join_context(
            child,
            extra_child,
            child_row2,
            exclude_parent=measure.source_table,
        )
        self._copy_group_values(child, extra_child, child_row2)
        self._ensure_parent_chain(d1, child, extra_child)
        self._append_once(d1, child, extra_child)
        self._ensure_filter_context(d0, measure.source_table)
        self._ensure_filter_context(d1, measure.source_table)
        self._ensure_required_tables(d0, measure.source_table)
        self._ensure_required_tables(d1, measure.source_table)
        pair = self._with_filters(
            TestPair("ART", measure, d0, d1, meta={"child_table": child})
        )
        self._stabilize_contract_source_domain(pair)
        return pair

    def _copy_query_join_context(
        self,
        table: str,
        target: dict[str, Any],
        source: dict[str, Any],
        *,
        exclude_parent: str,
    ) -> None:
        """Keep query-used non-target FK roles fixed across a replication."""

        copied: set[str] = set()
        for fk in self.registry.tables[table].foreign_keys:
            if fk.ref_table == exclude_parent:
                continue
            if not self._fk_is_used_by_query(table, fk.ref_table, fk):
                continue
            for column in fk.columns:
                target[column] = source[column]
                copied.add(column)

        # Some benchmark registries omit declared FKs that are nevertheless
        # explicit in the candidate's JOIN predicate (for example
        # Match.country_id = Country.id).  The SQL join equality is sufficient
        # to identify the child-side context column that must stay fixed.
        for left_ref, right_ref in self.join_column_pairs:
            if "." not in left_ref or "." not in right_ref:
                continue
            left_table, left_column = left_ref.split(".", 1)
            right_table, right_column = right_ref.split(".", 1)
            left_table = self.registry.resolve_table_name(left_table)
            right_table = self.registry.resolve_table_name(right_table)
            if left_table == table and right_table != exclude_parent:
                column = self.registry.resolve_column_name(table, left_column)
            elif right_table == table and left_table != exclude_parent:
                column = self.registry.resolve_column_name(table, right_column)
            else:
                continue
            if column in target and column not in copied:
                target[column] = source[column]

    def _copy_group_values(
        self,
        table: str,
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        """Keep a replicated row in the same SQL output group.

        A metamorphic replication test must perturb multiplicity within an
        existing group.  Letting generated values for a child grouping column
        differ can instead create a new output group, which is not evidence
        that the aggregate at the original group changed.
        """
        for ref in self.group_columns:
            if "." not in ref:
                column = self.registry.resolve_column_name(table, ref)
                containing = [
                    relation
                    for relation in self.required_tables
                    if relation in self.registry.tables
                    and self.registry.resolve_column_name(relation, ref)
                    in self.registry.tables[relation].columns
                ]
                if len(containing) != 1 or containing[0] != table:
                    continue
            else:
                ref_table, raw_column = ref.split(".", 1)
                if self.registry.resolve_table_name(ref_table) != table:
                    continue
                column = self.registry.resolve_column_name(table, raw_column)
            if column in source:
                target[column] = source[column]

    def _stabilize_contract_source_domain(self, pair: TestPair) -> None:
        """Keep FK-support parents from changing the contract population.

        Multi-role relations such as ``Friend(student_id, friend_id)`` and the
        Formula-1 ``Match`` table can reference the same entity table through
        several FKs. A D1 detail row may require an additional parent for an
        FK that the candidate does not join. Copying that isolated support row
        into D0 keeps the entity domain (and contract truth) identical while
        preserving the intended replication difference on the query-used FK.
        """
        table = pair.measure.source_table
        schema = self.registry.tables[table]
        key = schema.primary_key or schema.grain
        if not key:
            return
        rows0 = pair.d0.get(table) or []
        rows1 = pair.d1.get(table) or []
        by_key0 = {tuple(row.get(column) for column in key): row for row in rows0}
        by_key1 = {tuple(row.get(column) for column in key): row for row in rows1}
        for row_key, row in by_key1.items():
            if row_key not in by_key0:
                rows0.append(dict(row))
        for row_key, row in by_key0.items():
            if row_key not in by_key1:
                rows1.append(dict(row))

    def _build_tst(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        group_table = self.contract.output_group.table
        group_col = self.contract.output_group.column
        source_table = measure.source_table
        d0 = self._empty_rows()
        d1 = self._empty_rows()

        if (
            measure.ranking_metric_semantic == "SUM_ONCE_PER_GRAIN"
            and self._tst_needs_sum_decomposition(measure)
        ):
            return self._build_tst_sum_decomposition(plan)

        if source_table == group_table:
            if self._is_unique_group(source_table, group_col) and measure.source_column and measure.source_column != group_col:
                group_a, group_b = self._tst_group_values(group_table, group_col)
                row_a = self._base_row(source_table, index=1)
                row_b_d0 = self._base_row(source_table, index=2)
                row_b_d1 = self._base_row(source_table, index=3)
                row_a[group_col] = group_a
                row_a[measure.source_column] = 100
                row_b_d0[group_col] = group_b
                row_b_d0[measure.source_column] = 1
                row_b_d1[group_col] = group_b
                row_b_d1[measure.source_column] = 2
                self._ensure_parent_chain(d0, source_table, row_a)
                self._ensure_parent_chain(d0, source_table, row_b_d0)
                self._append_once(d0, source_table, row_a)
                self._append_once(d0, source_table, row_b_d0)
                self._ensure_parent_chain(d1, source_table, row_a)
                self._ensure_parent_chain(d1, source_table, row_b_d1)
                self._append_once(d1, source_table, dict(row_a))
                self._append_once(d1, source_table, row_b_d1)
                self._ensure_filter_context(d0, measure.source_table)
                self._ensure_filter_context(d1, measure.source_table)
                self._ensure_required_tables(d0, measure.source_table)
                self._ensure_required_tables(d1, measure.source_table)
                return self._with_filters(TestPair("TST", measure, d0, d1, meta={"expected_top_group": group_a}))
            group_a, group_b = self._tst_group_values(group_table, group_col)
            self._append_grouped_source_rows(d0, source_table, group_col, group_a, 1, 3)
            self._append_grouped_source_rows(d0, source_table, group_col, group_b, 101, 1)
            self._append_grouped_source_rows(d1, source_table, group_col, group_a, 1, 2)
            self._append_grouped_source_rows(d1, source_table, group_col, group_b, 101, 3)
            return self._with_filters(TestPair("TST", measure, d0, d1, meta={"expected_top_group": group_a}))

        source_to_group = self.registry.fk_to_parent(source_table, group_table)
        group_to_source = self.registry.fk_to_parent(group_table, source_table)
        if source_to_group:
            group_a = self._base_row(group_table, index=1)
            group_b = self._base_row(group_table, index=2)
            group_a_value, group_b_value = self._tst_group_values(group_table, group_col)
            group_a[group_col] = group_a_value
            group_b[group_col] = group_b_value
            self._append_child_rows_for_group(d0, source_table, group_table, group_a, 1, 3)
            self._append_child_rows_for_group(d0, source_table, group_table, group_b, 101, 1)
            self._append_child_rows_for_group(d1, source_table, group_table, group_a, 1, 2)
            self._append_child_rows_for_group(d1, source_table, group_table, group_b, 101, 3)
            return self._with_filters(TestPair("TST", measure, d0, d1, meta={"expected_top_group": group_a_value}))

        if group_to_source:
            raise ValueError("TST witness for group child of source is not implemented")
        raise ValueError("TST requires a direct join between source and output group")

    def _tst_needs_sum_decomposition(self, measure: MeasureContract) -> bool:
        """Return false when equality filters already fix one row per group."""
        if self.contract.output_group.table != measure.source_table:
            return True
        group_col = self.contract.output_group.column.casefold()
        remaining = [
            ref
            for ref in measure.base_grain
            if "." in ref
            and ref.split(".", 1)[0].casefold()
            == measure.source_table.casefold()
            and ref.split(".", 1)[1].casefold() != group_col
        ]
        if not remaining:
            return False
        exact_filters: set[str] = set()
        for item in self.filters:
            if item.operator not in {"=", "EQ"}:
                continue
            ref = item.source
            if "." not in ref:
                column = self.registry.resolve_column_name(
                    measure.source_table, ref
                )
                if column in self.registry.tables[measure.source_table].columns:
                    ref = f"{measure.source_table}.{column}"
            exact_filters.add(ref.casefold())
        return not all(ref.casefold() in exact_filters for ref in remaining)

    def _build_tst_sum_decomposition(self, plan: TestPlan) -> TestPair:
        """Preserve a winning group SUM while flipping its largest detail row."""
        measure = plan.measure
        source_table = measure.source_table
        group_table = self.contract.output_group.table
        group_col = self.contract.output_group.column
        metric_col = measure.source_column
        if not metric_col:
            raise ValueError("SUM-based TST requires source_column")
        d0 = self._empty_rows()
        d1 = self._empty_rows()
        group_a_value, group_b_value = self._tst_group_values(
            group_table, group_col
        )
        group_identity_col = self._tst_group_identity_column(group_table, group_col)
        ascending = (
            str((self.contract.ranking or {}).get("direction") or "DESC").upper()
            == "ASC"
        )
        source_to_group_path = self.registry.join_path(source_table, group_table)

        if source_table == group_table and not ascending:
            self._append_tst_source_row(
                d0, source_table, group_col, group_a_value, metric_col, 120, 1
            )
            self._append_tst_source_row(
                d0, source_table, group_col, group_b_value, metric_col, 100, 2
            )
            self._append_tst_source_row(
                d1, source_table, group_col, group_a_value, metric_col, 60, 1
            )
            self._append_tst_source_row(
                d1, source_table, group_col, group_a_value, metric_col, 60, 2
            )
            self._append_tst_source_row(
                d1, source_table, group_col, group_b_value, metric_col, 100, 3
            )
        elif source_table == group_table:
            # Keep A as the true lowest-SUM group in both D0 and D1.  The
            # DESC-oriented construction above made B the D0 winner under
            # ASC, so correct SUM queries were falsely reported as changing.
            self._append_tst_source_row(
                d0, source_table, group_col, group_a_value, metric_col, 50, 1
            )
            self._append_tst_source_row(
                d0, source_table, group_col, group_a_value, metric_col, 50, 2
            )
            self._append_tst_source_row(
                d0, source_table, group_col, group_b_value, metric_col, 120, 3
            )
            self._append_tst_source_row(
                d1, source_table, group_col, group_a_value, metric_col, 100, 1
            )
            self._append_tst_source_row(
                d1, source_table, group_col, group_b_value, metric_col, 60, 2
            )
            self._append_tst_source_row(
                d1, source_table, group_col, group_b_value, metric_col, 60, 3
            )
        else:
            directed_path = (
                source_to_group_path
                and len(source_to_group_path) > 2
                and all(
                    self.registry.fk_to_parent(child, parent)
                    for child, parent in zip(
                        source_to_group_path, source_to_group_path[1:]
                    )
                )
            )
            if directed_path:
                group_a = self._base_row(group_table, index=1)
                group_b = self._base_row(group_table, index=2)
                group_a[group_col] = group_a_value
                group_b[group_col] = group_b_value
                if not ascending:
                    values = ((d0, group_a, 120, 1), (d0, group_b, 100, 2),
                              (d1, group_a, 60, 1), (d1, group_a, 60, 2),
                              (d1, group_b, 100, 3))
                else:
                    values = ((d0, group_a, 50, 1), (d0, group_a, 50, 2),
                              (d0, group_b, 120, 3), (d1, group_a, 100, 1),
                              (d1, group_b, 60, 2), (d1, group_b, 60, 3))
                for rows, group_row, value, index in values:
                    self._append_tst_path_metric_row(
                        rows,
                        source_table,
                        source_to_group_path,
                        group_table,
                        group_row,
                        group_identity_col,
                        metric_col,
                        value,
                        index,
                    )
            elif not self.registry.fk_to_parent(source_table, group_table):
                raise ValueError(
                    "SUM-based TST requires source table to be the group table "
                    "or a directed child-to-parent join path"
                )
            else:
                group_a = self._base_row(group_table, index=1)
                group_b = self._base_row(group_table, index=2)
                group_a[group_col] = group_a_value
                group_b[group_col] = group_b_value
                if not ascending:
                    values = ((d0, group_a, 120, 1), (d0, group_b, 100, 2),
                              (d1, group_a, 60, 1), (d1, group_a, 60, 2),
                              (d1, group_b, 100, 3))
                else:
                    values = ((d0, group_a, 50, 1), (d0, group_a, 50, 2),
                              (d0, group_b, 120, 3), (d1, group_a, 100, 1),
                              (d1, group_b, 60, 2), (d1, group_b, 60, 3))
                for rows, group_row, value, index in values:
                    self._append_tst_child_metric_row(
                        rows, source_table, group_table, group_row, metric_col,
                        value, index, group_identity_col=group_identity_col,
                    )

        for rows in (d0, d1):
            self._ensure_filter_context(rows, source_table)
            self._ensure_required_tables(rows, source_table)
        return self._with_filters(
            TestPair(
                "TST",
                measure,
                d0,
                d1,
                meta={"expected_top_group": group_a_value},
            )
        )

    def _append_tst_source_row(
        self,
        rows: Rows,
        table: str,
        group_col: str,
        group_value: Any,
        metric_col: str,
        metric_value: float,
        index: int,
    ) -> None:
        row = self._base_row(table, index=index)
        row[group_col] = group_value
        row[metric_col] = metric_value
        self._ensure_parent_chain(rows, table, row)
        self._append_once(rows, table, row)

    def _append_tst_child_metric_row(
        self,
        rows: Rows,
        child: str,
        parent: str,
        parent_row: dict[str, Any],
        metric_col: str,
        metric_value: float,
        index: int,
        *,
        group_identity_col: str | None = None,
    ) -> None:
        linked_parent = parent_row
        group_col = group_identity_col or self.contract.output_group.column
        if not self._is_unique_group(parent, group_col):
            linked_parent = self._variant_row(parent, parent_row, index)
        self._ensure_parent_chain(rows, parent, linked_parent)
        self._append_once(rows, parent, linked_parent)
        row = self._base_row(child, index=index)
        row[metric_col] = metric_value
        self._link_child_to_parent(child, row, parent, linked_parent)
        self._ensure_parent_chain(rows, child, row)
        self._append_once(rows, child, row)

    def _tst_group_identity_column(self, group_table: str, fallback: str) -> str:
        """Use the SQL grouping key when the displayed output is non-unique."""
        for ref in self.group_columns:
            if "." not in ref:
                continue
            table, column = ref.split(".", 1)
            if self.registry.resolve_table_name(table) == group_table:
                return self.registry.resolve_column_name(group_table, column)
        return fallback

    def _append_tst_path_metric_row(
        self,
        rows: Rows,
        source_table: str,
        path: list[str],
        group_table: str,
        group_row: dict[str, Any],
        group_identity_col: str,
        metric_col: str,
        metric_value: float,
        index: int,
    ) -> None:
        """Materialize a source-to-group ancestor path for a TST SUM pair."""
        parent_row = group_row
        parent_table = group_table
        if not self._is_unique_group(group_table, group_identity_col):
            parent_row = self._variant_row(group_table, group_row, index)
        self._ensure_parent_chain(rows, group_table, parent_row)
        self._append_once(rows, group_table, parent_row)
        for child_table in reversed(path[:-1]):
            row = self._base_row(child_table, index=index)
            if child_table == source_table:
                row[metric_col] = metric_value
            self._link_child_to_parent(child_table, row, parent_table, parent_row)
            self._ensure_parent_chain(rows, child_table, row)
            self._append_once(rows, child_table, row)
            parent_table, parent_row = child_table, row

    def _build_synthetic_grain_fault(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        fault_col = self._synthetic_fault_column(measure)
        if not fault_col:
            if plan.test_type == "CGR":
                fault_col = self._first_base_grain_column(measure)
            elif plan.test_type in {"CRT", "GRT"}:
                return self._build_synthetic_grt_count_fault(plan)
        if not fault_col:
            raise ValueError(f"no synthetic fault column is available for {measure.source_table}")
        d0 = self._empty_rows()
        source_row = self._base_row(measure.source_table)
        if measure.source_column:
            source_row[measure.source_column] = 100
        source_row[fault_col] = self._synthetic_non_trigger_value(measure.source_table, fault_col)
        self._ensure_parent_chain(d0, measure.source_table, source_row)
        self._append_once(d0, measure.source_table, source_row)

        d1 = self._clone(d0)
        d1[measure.source_table][0][fault_col] = self._synthetic_trigger_value(measure.source_table, fault_col)
        self._ensure_filter_context(d0, measure.source_table)
        self._ensure_filter_context(d1, measure.source_table)
        self._ensure_required_tables(d0, measure.source_table)
        self._ensure_required_tables(d1, measure.source_table)
        return self._with_filters(
            TestPair(
                plan.test_type,
                measure,
                d0,
                d1,
                meta={"child_table": SYNTHETIC_FAULT_CHILD, "synthetic_fault_column": fault_col},
            )
        )

    def _build_synthetic_grt_count_fault(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        d0 = self._empty_rows()
        source_row = self._base_row(measure.source_table, index=1)
        source_row[measure.source_column] = 100
        self._ensure_parent_chain(d0, measure.source_table, source_row)
        self._append_once(d0, measure.source_table, source_row)

        d1 = self._empty_rows()
        source_row1 = self._base_row(measure.source_table, index=1)
        source_row2 = self._base_row(measure.source_table, index=2)
        for grain_ref in measure.base_grain:
            if "." not in grain_ref:
                continue
            grain_table, grain_col = grain_ref.split(".", 1)
            if grain_table != measure.source_table:
                continue
            typ = self.registry.tables[grain_table].columns[grain_col]
            source_row1[grain_col] = default_value(
                grain_table, grain_col, typ, index=1
            )
            source_row2[grain_col] = default_value(
                grain_table, grain_col, typ, index=2
            )
        source_row1[measure.source_column] = 40
        source_row2[measure.source_column] = 60
        self._ensure_parent_chain(d1, measure.source_table, source_row1)
        self._ensure_parent_chain(d1, measure.source_table, source_row2)
        self._append_once(d1, measure.source_table, source_row1)
        self._append_once(d1, measure.source_table, source_row2)
        self._ensure_filter_context(d0, measure.source_table)
        self._ensure_filter_context(d1, measure.source_table)
        self._ensure_required_tables(d0, measure.source_table)
        self._ensure_required_tables(d1, measure.source_table)
        return self._with_filters(
            TestPair(
                plan.test_type,
                measure,
                d0,
                d1,
                meta={"child_table": SYNTHETIC_FAULT_CHILD, "synthetic_fault": "count_sensitive_split"},
            )
        )

    def _build_synthetic_art_fault(self, plan: TestPlan) -> TestPair:
        measure = plan.measure
        fault_col = self._synthetic_fault_column(measure)
        d0 = self._empty_rows()

        source_row1 = self._base_row(measure.source_table, index=1)
        source_row2 = self._base_row(measure.source_table, index=2)
        source_row1[measure.source_column] = 100
        source_row2[measure.source_column] = 200
        source_row1[fault_col] = self._synthetic_non_trigger_value(measure.source_table, fault_col)
        source_row2[fault_col] = self._synthetic_non_trigger_value(measure.source_table, fault_col)
        self._ensure_parent_chain(d0, measure.source_table, source_row1)
        self._ensure_parent_chain(d0, measure.source_table, source_row2)
        self._append_once(d0, measure.source_table, source_row1)
        self._append_once(d0, measure.source_table, source_row2)

        d1 = self._clone(d0)
        d1[measure.source_table][-1][fault_col] = self._synthetic_trigger_value(measure.source_table, fault_col)
        self._ensure_filter_context(d0, measure.source_table)
        self._ensure_filter_context(d1, measure.source_table)
        self._ensure_required_tables(d0, measure.source_table)
        self._ensure_required_tables(d1, measure.source_table)
        return self._with_filters(
            TestPair(
                "ART",
                measure,
                d0,
                d1,
                meta={"child_table": SYNTHETIC_FAULT_CHILD, "synthetic_fault_column": fault_col},
            )
        )

    def _synthetic_fault_column(self, measure: MeasureContract) -> str | None:
        schema = self.registry.tables[measure.source_table]
        blocked = {measure.source_column}
        blocked.update(ref.split(".", 1)[1] for ref in measure.base_grain if "." in ref)
        foreign_key_columns = {
            column
            for fk in schema.foreign_keys
            for column in fk.columns
        }
        unique_columns = {
            column
            for columns in [schema.primary_key, *schema.unique]
            for column in columns
        }
        candidates = [
            col
            for col in schema.columns
            if col not in blocked
            and col not in foreign_key_columns
            and col not in unique_columns
        ]
        if not candidates:
            candidates = [
                col
                for col in schema.columns
                if col not in blocked and col not in unique_columns
            ]
        return candidates[0] if candidates else None

    def _is_unique_group(self, table: str, column: str) -> bool:
        schema = self.registry.tables[table]
        if schema.primary_key == [column]:
            return True
        return any(unique_cols == [column] for unique_cols in schema.unique)

    def _first_base_grain_column(self, measure: MeasureContract) -> str | None:
        for grain_ref in measure.base_grain:
            if "." not in grain_ref:
                continue
            table, column = grain_ref.split(".", 1)
            if table == measure.source_table and column in self.registry.tables[table].columns:
                return column
        return None

    def _synthetic_entity_fault_column(self, entity_table: str, entity_col: str) -> str:
        schema = self.registry.tables[entity_table]
        foreign_key_columns = {
            column
            for fk in schema.foreign_keys
            for column in fk.columns
        }
        unique_columns = {
            column
            for columns in [schema.primary_key, *schema.unique]
            for column in columns
        }
        candidates = [
            col
            for col in schema.columns
            if col != entity_col
            and col not in foreign_key_columns
            and col not in unique_columns
        ]
        if not candidates:
            candidates = [
                col
                for col in schema.columns
                if col != entity_col and col not in unique_columns
            ]
        return candidates[0] if candidates else entity_col

    def _synthetic_trigger_value(self, table: str, column: str) -> Any:
        return self._synthetic_literal_value(table, column, triggered=True)

    def _synthetic_non_trigger_value(self, table: str, column: str) -> Any:
        return self._synthetic_literal_value(table, column, triggered=False)

    def _synthetic_literal_value(self, table: str, column: str, *, triggered: bool) -> Any:
        typ = self.registry.tables[table].columns[column].lower()
        if "date" in typ or "time" in typ:
            return "2099-01-01" if triggered else "2000-01-01"
        if any(x in typ for x in ("int", "decimal", "numeric", "real", "float", "double")):
            return 999001 if triggered else 1
        return "__grain_fault_trigger__" if triggered else "__grain_fault_base__"

    def _tst_group_values(self, table: str, column: str) -> tuple[Any, Any]:
        typ = self.registry.tables[table].columns[column].lower()
        if any(x in typ for x in ("int", "decimal", "numeric", "real", "float", "double")):
            return 100, 200
        return "A", "B"

    def _append_grouped_source_rows(
        self,
        rows: Rows,
        table: str,
        group_col: str,
        group_value: str,
        start_index: int,
        count: int,
    ) -> None:
        for offset in range(count):
            row = self._base_row(table, index=start_index + offset)
            row[group_col] = group_value
            self._ensure_parent_chain(rows, table, row)
            self._append_once(rows, table, row)

    def _append_child_rows_for_group(
        self,
        rows: Rows,
        child: str,
        parent: str,
        parent_row: dict[str, Any],
        start_index: int,
        count: int,
    ) -> None:
        for offset in range(count):
            parent_row_i = self._variant_row(parent, parent_row, start_index + offset)
            self._ensure_parent_chain(rows, parent, parent_row_i)
            self._append_once(rows, parent, parent_row_i)
            child_row = self._base_row(child, index=start_index + offset)
            self._link_child_to_parent(child, child_row, parent, parent_row_i)
            self._ensure_parent_chain(rows, child, child_row)
            self._append_once(rows, child, child_row)

    def _variant_row(self, table: str, row: dict[str, Any], index: int) -> dict[str, Any]:
        variant = dict(row)
        group_col = self.contract.output_group.column if table == self.contract.output_group.table else None
        schema = self.registry.tables[table]
        variant_columns = {
            column
            for columns in [schema.primary_key, *schema.unique]
            for column in columns
        }
        for col in variant_columns:
            if col == group_col:
                continue
            variant[col] = default_value(table, col, schema.columns[col], index=index)
        return variant

    def _fk_col(self, child: str, parent: str) -> str:
        fk = self.registry.fk_to_parent(child, parent)
        if not fk:
            raise ValueError(f"No FK from {child} to {parent}")
        return fk.columns[0]

    def _link_child_to_parent(self, child: str, child_row: dict[str, Any], parent: str, parent_row: dict[str, Any]) -> None:
        matched = False
        candidates = [
            fk
            for fk in self.registry.tables[child].foreign_keys
            if fk.ref_table == parent
        ]
        selected = [
            fk
            for fk in candidates
            if self._fk_is_used_by_query(child, parent, fk)
        ]
        if not selected:
            selected = candidates
        if len(selected) > 1 and self.join_column_pairs:
            selected = selected[:1]
        for fk in selected:
            for child_col, parent_col in zip(fk.columns, fk.ref_columns):
                child_row[child_col] = parent_row[parent_col]
            matched = True
        if not matched:
            if self._link_sibling_to_parent_key(child, child_row, parent, parent_row):
                return
            # Allow linking through direct same-named entity key in simple event tables.
            return

    def _fk_is_used_by_query(
        self,
        child: str,
        parent: str,
        fk: ForeignKey,
    ) -> bool:
        if not self.join_column_pairs:
            return False
        expected = {
            frozenset(
                {
                    f"{child}.{child_col}".casefold(),
                    f"{parent}.{parent_col}".casefold(),
                }
            )
            for child_col, parent_col in zip(fk.columns, fk.ref_columns)
        }
        actual = {
            frozenset({left.casefold(), right.casefold()})
            for left, right in self.join_column_pairs
        }
        return bool(expected) and expected <= actual

    def _link_sibling_to_parent_key(
        self,
        child: str,
        child_row: dict[str, Any],
        parent: str,
        parent_row: dict[str, Any],
    ) -> bool:
        for child_fk in self.registry.tables[child].foreign_keys:
            for parent_fk in self.registry.tables[parent].foreign_keys:
                if child_fk.ref_table != parent_fk.ref_table:
                    continue
                if child_fk.ref_columns != parent_fk.ref_columns:
                    continue
                for child_col, parent_col in zip(child_fk.columns, parent_fk.columns):
                    child_row[child_col] = parent_row[parent_col]
                return True
        return False

    def _with_filters(self, pair: TestPair) -> TestPair:
        for rows in (pair.d0, pair.d1):
            if self.filters:
                self._apply_filters(rows, pair)
                self._propagate_single_parent_keys(rows)
                self._apply_filters(rows, pair)
            self._dedupe_key_rows(rows)
            self._complete_missing_parent_rows(rows)
            if self.filters:
                # Parent completion can add rows after the earlier filter
                # pass (for example Match.country_id -> Country). Those rows
                # must receive the same query predicates or a valid source
                # entity may disappear from the synthetic join result.
                self._apply_filters(rows, pair)
                self._propagate_single_parent_keys(rows)
                self._complete_missing_parent_rows(rows)
        return pair

    def _complete_missing_parent_rows(self, rows: Rows) -> None:
        """Complete FK parents after filters or synthetic perturbations change keys."""
        changed = True
        while changed:
            changed = False
            for child_name, child_schema in self.registry.tables.items():
                child_rows = rows.get(child_name) or []
                if not child_rows:
                    continue
                for fk in child_schema.foreign_keys:
                    parent_rows = rows.get(fk.ref_table) or []
                    parent_keys = {
                        tuple(parent.get(column) for column in fk.ref_columns)
                        for parent in parent_rows
                    }
                    for child_row in child_rows:
                        key = tuple(child_row.get(column) for column in fk.columns)
                        if any(value is None for value in key) or key in parent_keys:
                            continue
                        parent = self._base_row(
                            fk.ref_table,
                            index=len(parent_rows) + 1,
                        )
                        for parent_col, value in zip(fk.ref_columns, key):
                            parent[parent_col] = value
                        self._ensure_parent_chain(rows, fk.ref_table, parent)
                        before = len(rows[fk.ref_table])
                        self._append_once(rows, fk.ref_table, parent)
                        if len(rows[fk.ref_table]) > before:
                            changed = True
                            parent_rows = rows[fk.ref_table]
                            parent_keys.add(key)

    def _apply_conditional_aggregate_stress(self, pair: TestPair) -> None:
        if not self.conditional_filters:
            return
        if (
            pair.test_type == "GRT"
            and self._add_grt_contrast_source_for_conditional(pair)
        ):
            return
        if pair.test_type != "DET":
            child_table = pair.meta.get("child_table")
            if not child_table:
                return
            for rows in (pair.d0, pair.d1):
                self._apply_filter_list(rows, pair, self.conditional_filters)
            if (
                pair.test_type == "CGR"
                and self._add_cgr_contrast_source_for_conditional(
                    pair, child_table
                )
            ):
                return
            self._make_duplicate_event_fail_first_condition(pair, child_table)
            return
        for rows in (pair.d0, pair.d1):
            self._apply_filter_list(rows, pair, self.conditional_filters)
        event_table = pair.meta.get("event_table")
        if not event_table or event_table not in self.registry.tables:
            return
        event_rows = pair.d1.get(event_table) or []
        if len(event_rows) < 2:
            return
        entity_filters = self._conditional_filters_for_entity(pair)
        if entity_filters:
            self._add_det_contrast_entity_for_conditional(pair, entity_filters)
            return
        self._make_duplicate_event_fail_first_condition(pair, event_table)

    def _add_grt_contrast_source_for_conditional(
        self,
        pair: TestPair,
    ) -> bool:
        """Expose non-uniform fan-out across two conditional metric classes.

        Growth and period-comparison SQL often contains two conditional SUMs.
        Replicating the only source row scales numerator and denominator
        together, hiding a real fan-out error.  Add a second source-grain row
        on an independent parent context and place the two rows in different
        conditional classes.  D0 and D1 retain identical source rows and
        contract values; only the first class is multiplied by the query join.
        """

        source_table = pair.measure.source_table
        candidates: list[tuple[FilterSignature, str]] = []
        for item in self.conditional_filters:
            targets = [
                (table, column)
                for table, column in self._filter_targets(item.source, pair.d0)
                if table == source_table
            ]
            if (
                len(targets) == 1
                and item.operator in {"=", "EQ", "YEAR_EQ"}
            ):
                candidates.append((item, targets[0][1]))
        selected: tuple[FilterSignature, FilterSignature, str] | None = None
        for index, (left, left_column) in enumerate(candidates):
            for right, right_column in candidates[index + 1 :]:
                if (
                    left_column == right_column
                    and left.value != right.value
                ):
                    selected = (left, right, left_column)
                    break
            if selected:
                break
        if selected is None:
            return False
        left, right, class_column = selected
        if not pair.d0.get(source_table) or not pair.d1.get(source_table):
            return False

        original = pair.d0[source_table][0]
        contrast = self._variant_row(
            source_table,
            original,
            index=len(pair.d0[source_table]) + 3,
        )
        for grain_ref in pair.measure.base_grain:
            if "." not in grain_ref:
                continue
            grain_table, grain_column = grain_ref.split(".", 1)
            grain_table = self.registry.resolve_table_name(grain_table)
            if grain_table != source_table:
                continue
            grain_column = self.registry.resolve_column_name(
                source_table,
                grain_column,
            )
            contrast[grain_column] = default_value(
                source_table,
                grain_column,
                self.registry.tables[source_table].columns[grain_column],
                index=len(pair.d0[source_table]) + 3,
            )
        parent_fk = next(
            iter(self.registry.tables[source_table].foreign_keys),
            None,
        )
        if parent_fk is None:
            return False
        parent = self._base_row(parent_fk.ref_table, index=3)
        self._link_child_to_parent(
            source_table,
            contrast,
            parent_fk.ref_table,
            parent,
        )
        if pair.measure.source_column:
            contrast[pair.measure.source_column] = original[
                pair.measure.source_column
            ]

        left_value = self._satisfying_value(
            self.registry.tables[source_table].columns[class_column],
            left.operator,
            left.value,
        )
        right_value = self._satisfying_value(
            self.registry.tables[source_table].columns[class_column],
            right.operator,
            right.value,
        )
        for rows in (pair.d0, pair.d1):
            rows[source_table][0][class_column] = left_value
            row = dict(contrast)
            row[class_column] = right_value
            self._ensure_parent_chain(rows, source_table, row)
            self._append_once(rows, source_table, row)
            # Materialize query relations for every parent context, not only
            # for the already-populated first source row.
            for _ in range(len(self.required_tables) + 1):
                for table in sorted(self.required_tables):
                    if (
                        table in self.registry.tables
                        and table != source_table
                    ):
                        self._ensure_child_rows_for_existing_parents(
                            rows,
                            table,
                        )
                self._complete_missing_parent_rows(rows)
            self._apply_filters(rows, pair)
            rows[source_table][0][class_column] = left_value
            rows[source_table][-1][class_column] = right_value
        return True

    def _add_cgr_contrast_source_for_conditional(
        self,
        pair: TestPair,
        child_table: str,
    ) -> bool:
        """Add a second semantic class for conditional count ratios.

        With one source row, duplicating a sibling row scales both numerator
        and denominator equally. A contrast source on a different shared
        parent makes non-uniform reweighting observable while preserving the
        number of source-grain entities in D0 and D1.
        """
        source_table = pair.measure.source_table
        source_filters = [
            item
            for item in self.conditional_filters
            if any(
                table == source_table
                for table, _ in self._filter_targets(item.source, pair.d0)
            )
        ]
        if not source_filters or child_table not in self.registry.tables:
            return False
        shared = next(
            (
                (source_fk, child_fk)
                for source_fk in self.registry.tables[source_table].foreign_keys
                for child_fk in self.registry.tables[child_table].foreign_keys
                if source_fk.ref_table == child_fk.ref_table
                and source_fk.ref_columns == child_fk.ref_columns
            ),
            None,
        )
        if not shared or not pair.d0.get(source_table):
            return False
        source_fk, child_fk = shared
        parent_table = source_fk.ref_table
        parent = self._base_row(parent_table, index=3)
        source = self._variant_row(
            source_table, pair.d0[source_table][0], index=3
        )
        for grain_ref in pair.measure.base_grain:
            if "." not in grain_ref:
                continue
            grain_table, grain_col = grain_ref.split(".", 1)
            if grain_table != source_table:
                continue
            source[grain_col] = default_value(
                grain_table,
                grain_col,
                self.registry.tables[grain_table].columns[grain_col],
                index=3,
            )
        child = self._base_row(child_table, index=3)
        for source_col, parent_col in zip(
            source_fk.columns, source_fk.ref_columns
        ):
            source[source_col] = parent[parent_col]
        for child_col, parent_col in zip(
            child_fk.columns, child_fk.ref_columns
        ):
            child[child_col] = parent[parent_col]

        for rows in (pair.d0, pair.d1):
            self._ensure_parent_chain(rows, parent_table, dict(parent))
            self._append_once(rows, parent_table, dict(parent))
            self._ensure_parent_chain(rows, source_table, dict(source))
            self._append_once(rows, source_table, dict(source))
            self._ensure_parent_chain(rows, child_table, dict(child))
            self._append_once(rows, child_table, dict(child))
            self._ensure_required_tables(rows, source_table)
            self._complete_missing_parent_rows(rows)
            self._apply_filter_list(rows, pair, self.filters)
            contrast = rows[source_table][-1]
            for item in source_filters:
                targets = [
                    target
                    for target in self._filter_targets(item.source, rows)
                    if target[0] == source_table
                ]
                if not targets:
                    continue
                _, column = targets[0]
                contrast[column] = self._non_satisfying_value(
                    self.registry.tables[source_table].columns[column],
                    item.operator,
                    item.value,
                )
        return True

    def _make_duplicate_event_fail_first_condition(
        self,
        pair: TestPair,
        event_table: str,
    ) -> None:
        event_rows = pair.d1.get(event_table) or []
        if len(event_rows) < 2:
            return
        duplicate_row = event_rows[1]
        for filter_sig in self.conditional_filters:
            targets = self._filter_targets(filter_sig.source, pair.d1)
            event_targets = [(table, column) for table, column in targets if table == event_table]
            if not event_targets:
                continue
            table, column = event_targets[0]
            duplicate_row[column] = self._non_satisfying_value(
                self.registry.tables[table].columns[column],
                filter_sig.operator,
                filter_sig.value,
            )
            return

    def _has_single_equality_entity_class(
        self,
        entity_filters: list[FilterSignature],
    ) -> bool:
        equality_values = {
            item.value
            for item in entity_filters
            if item.operator in {"=", "EQ"}
        }
        return len(equality_values) == 1

    def _conditional_filters_for_entity(self, pair: TestPair) -> list[FilterSignature]:
        if not pair.measure.entity_key:
            return []
        entity_table, _ = pair.measure.entity_key.split(".", 1)
        return [
            filter_sig
            for filter_sig in self.conditional_filters
            if any(
                table == entity_table
                for table, _ in self._filter_targets(filter_sig.source, pair.d1)
            )
        ]

    def _add_det_contrast_entity_for_conditional(
        self,
        pair: TestPair,
        entity_filters: list[FilterSignature],
    ) -> None:
        if not pair.measure.entity_key or not pair.meta.get("event_entity_cols"):
            return
        entity_table, entity_col = pair.measure.entity_key.split(".", 1)
        event_table = pair.meta.get("event_table")
        if not event_table or event_table == entity_table:
            return

        entity_row = self._variant_row(
            entity_table,
            pair.d0[entity_table][0],
            index=3,
        )
        entity_row[entity_col] = default_value(
            entity_table,
            entity_col,
            self.registry.tables[entity_table].columns[entity_col],
            index=3,
        )
        self._ensure_parent_chain(pair.d0, entity_table, entity_row)
        self._ensure_parent_chain(pair.d1, entity_table, dict(entity_row))
        self._append_once(pair.d0, entity_table, entity_row)
        self._append_once(pair.d1, entity_table, dict(entity_row))

        event_row_d0 = self._base_row(event_table, index=3)
        event_row_d1 = self._base_row(event_table, index=3)
        self._link_child_to_parent(event_table, event_row_d0, entity_table, entity_row)
        self._link_child_to_parent(event_table, event_row_d1, entity_table, entity_row)
        self._append_once(pair.d0, event_table, event_row_d0)
        self._append_once(pair.d1, event_table, event_row_d1)

        for rows in (pair.d0, pair.d1):
            for table in sorted(self.required_tables):
                if table in self.registry.tables:
                    self._ensure_child_rows_for_existing_parents(rows, table)
            self._ensure_required_tables(rows, event_table)
            self._complete_missing_parent_rows(rows)
            self._apply_filter_list(rows, pair, self.filters)
            self._apply_filter_list(rows, pair, self.conditional_filters)

        for rows, contrast_event in (
            (pair.d0, event_row_d0),
            (pair.d1, event_row_d1),
        ):
            contrast_row = next(
                row
                for row in rows[entity_table]
                if row.get(entity_col) == entity_row.get(entity_col)
            )
            grouped_entity = self._group_filters_by_target(entity_filters, rows, entity_table)
            for column, filters in grouped_entity.items():
                equality_values = list(
                    dict.fromkeys(
                        item.value
                        for item in filters
                        if item.operator in {"=", "EQ"}
                    )
                )
                if len(equality_values) >= 2:
                    # Preserve two semantic classes explicitly. Applying the
                    # CASE predicates sequentially would otherwise leave both
                    # D0/D1 base entities in the final class (e.g. F after
                    # M,F), so duplicating that class may not cross a boolean
                    # comparison boundary such as male_count > female_count.
                    rows[entity_table][0][column] = equality_values[0]
                    contrast_row[column] = equality_values[1]
                    continue
                current = rows[entity_table][0][column]
                alternatives = [
                    item.value
                    for item in filters
                    if item.operator in {"=", "EQ"} and item.value != current
                ]
                if alternatives:
                    contrast_row[column] = alternatives[0]
                else:
                    item = filters[0]
                    contrast_row[column] = self._non_satisfying_value(
                        self.registry.tables[entity_table].columns[column],
                        item.operator,
                        item.value,
                    )

            event_filters = [
                item
                for item in self.conditional_filters
                if any(table == event_table for table, _ in self._filter_targets(item.source, rows))
            ]
            grouped_event = self._group_filters_by_target(event_filters, rows, event_table)
            for column, filters in grouped_event.items():
                joint_value = self._joint_satisfying_value(
                    self.registry.tables[event_table].columns[column],
                    filters,
                )
                for event_row in rows.get(event_table) or []:
                    event_row[column] = joint_value

    def _group_filters_by_target(
        self,
        filters: list[FilterSignature],
        rows: Rows,
        target_table: str,
    ) -> dict[str, list[FilterSignature]]:
        grouped: dict[str, list[FilterSignature]] = {}
        for item in filters:
            for table, column in self._filter_targets(item.source, rows):
                if table == target_table:
                    grouped.setdefault(column, []).append(item)
        return grouped

    def _joint_satisfying_value(
        self,
        type_name: str,
        filters: list[FilterSignature],
    ) -> Any:
        upper = [
            item.value
            for item in filters
            if item.operator in {"<", "<=", "LT", "LTE"}
            and isinstance(item.value, (int, float))
        ]
        lower = [
            item.value
            for item in filters
            if item.operator in {">", ">=", "GT", "GTE"}
            and isinstance(item.value, (int, float))
        ]
        if upper:
            bound = min(upper)
            strict = any(
                item.operator in {"<", "LT"} and item.value == bound
                for item in filters
            )
            return bound - 1 if strict else bound
        if lower:
            bound = max(lower)
            strict = any(
                item.operator in {">", "GT"} and item.value == bound
                for item in filters
            )
            return bound + 1 if strict else bound
        item = filters[0]
        return self._satisfying_value(type_name, item.operator, item.value)

    def _apply_filter_list(
        self,
        rows: Rows,
        pair: TestPair,
        filters: list[FilterSignature],
    ) -> None:
        for filter_sig in filters:
            for table, column in self._filter_targets(filter_sig.source, rows):
                values = self._satisfying_values_for_rows(table, column, filter_sig, rows[table], pair)
                for row, value in zip(rows[table], values):
                    row[column] = value

    def _ensure_filter_context(self, rows: Rows, source_table: str) -> None:
        source_rows = rows.get(source_table) or []
        if not source_rows:
            return
        for filter_sig in self.filters:
            if "." not in filter_sig.source:
                continue
            table, _ = filter_sig.source.split(".", 1)
            if table not in self.registry.tables or rows.get(table):
                continue
            for index, source_row in enumerate(source_rows, start=1):
                filter_row = self._base_row(table, index=index)
                self._link_child_to_parent(table, filter_row, source_table, source_row)
                self._link_child_to_parent(source_table, source_row, table, filter_row)
                if self._has_equivalent_source_context(
                    rows.get(table) or [],
                    table,
                    filter_row,
                    source_table,
                ):
                    continue
                self._ensure_parent_chain(rows, table, filter_row)
                self._append_once(rows, table, filter_row)

    def _has_equivalent_source_context(
        self,
        existing_rows: list[dict[str, Any]],
        table: str,
        candidate: dict[str, Any],
        source_table: str,
    ) -> bool:
        shared_columns: set[str] = set()
        source_parent_refs = {
            (fk.ref_table, tuple(fk.ref_columns))
            for fk in self.registry.tables[source_table].foreign_keys
        }
        for fk in self.registry.tables[table].foreign_keys:
            if (fk.ref_table, tuple(fk.ref_columns)) in source_parent_refs:
                shared_columns.update(fk.columns)
            if fk.ref_table == source_table:
                shared_columns.update(fk.columns)
        if not shared_columns:
            return False
        return any(
            all(row.get(column) == candidate.get(column) for column in shared_columns)
            for row in existing_rows
        )

    def _ensure_required_tables(self, rows: Rows, source_table: str) -> None:
        required = {table for table in self.required_tables if table in self.registry.tables}
        if not required:
            return
        changed = True
        while changed:
            changed = False
            for table in sorted(required):
                if rows.get(table):
                    continue
                if self._ensure_child_rows_for_existing_parents(rows, table):
                    changed = True
                    if rows.get(table):
                        continue
                row = self._row_from_existing_parent(table, rows)
                if row is None:
                    row = self._row_from_existing_child(table, rows)
                if row is None:
                    row = self._row_from_source_context(table, source_table, rows)
                if row is None:
                    row = self._row_from_any_join_context(table, rows)
                if row is None:
                    continue
                self._ensure_parent_chain(rows, table, row)
                before = len(rows[table])
                self._append_once(rows, table, row)
                changed = changed or len(rows[table]) > before

    def _row_from_any_join_context(
        self,
        table: str,
        rows: Rows,
    ) -> dict[str, Any] | None:
        """Materialize a relation from any populated direct or sibling peer."""
        for peer, peer_rows in rows.items():
            if (
                peer == table
                or not peer_rows
                or peer not in self.registry.tables
            ):
                continue
            direct = bool(
                self.registry.fk_to_parent(table, peer)
                or self.registry.fk_to_parent(peer, table)
            )
            table_parents = {
                (fk.ref_table, tuple(fk.ref_columns))
                for fk in self.registry.tables[table].foreign_keys
            }
            peer_parents = {
                (fk.ref_table, tuple(fk.ref_columns))
                for fk in self.registry.tables[peer].foreign_keys
            }
            if not direct and not table_parents.intersection(peer_parents):
                continue
            row = self._base_row(table)
            self._link_child_to_parent(table, row, peer, peer_rows[0])
            self._link_child_to_parent(peer, peer_rows[0], table, row)
            return row
        return None

    def _ensure_child_rows_for_existing_parents(self, rows: Rows, table: str) -> bool:
        changed = False
        schema = self.registry.tables[table]
        for fk in schema.foreign_keys:
            parent_rows = rows.get(fk.ref_table) or []
            if not parent_rows:
                continue
            for index, parent_row in enumerate(parent_rows, start=1):
                if self._has_child_for_parent(rows.get(table) or [], fk, parent_row):
                    continue
                row = self._base_row(table, index=index)
                for child_col, parent_col in zip(fk.columns, fk.ref_columns):
                    row[child_col] = parent_row[parent_col]
                existing_keys = {
                    tuple(item.get(column) for column in schema.primary_key)
                    for item in rows.get(table) or []
                }
                if schema.primary_key and tuple(row.get(column) for column in schema.primary_key) in existing_keys:
                    # A replicated child may already occupy the parent's
                    # enumeration index. _append_once would silently discard
                    # this missing context, potentially removing a qualifying
                    # contrast unit from only one side of the witness.
                    bound_columns = set(fk.columns)
                    key_columns = {
                        column
                        for key in [schema.primary_key, *schema.unique]
                        for column in key
                    } - bound_columns
                    start = len(rows.get(table) or []) + 1
                    for fresh_index in range(start, start + len(existing_keys) + 1):
                        fresh = self._base_row(table, index=fresh_index)
                        candidate = dict(row)
                        for column in key_columns:
                            candidate[column] = fresh[column]
                        if not any(
                            all(candidate.get(column) is not None for column in key)
                            and any(
                                all(item.get(column) == candidate.get(column) for column in key)
                                for item in rows.get(table) or []
                            )
                            for key in [schema.primary_key, *schema.unique]
                            if key
                        ):
                            row = candidate
                            break
                    else:
                        raise ValueError(f"cannot allocate a distinct support key for {table} -> {fk.ref_table}")
                before = len(rows[table])
                self._append_once(rows, table, row)
                changed = changed or len(rows[table]) > before
        return changed

    def _has_child_for_parent(
        self,
        child_rows: list[dict[str, Any]],
        fk: Any,
        parent_row: dict[str, Any],
    ) -> bool:
        return any(
            all(child_row[child_col] == parent_row[parent_col] for child_col, parent_col in zip(fk.columns, fk.ref_columns))
            for child_row in child_rows
        )

    def _row_from_existing_parent(self, table: str, rows: Rows) -> dict[str, Any] | None:
        for fk in self.registry.tables[table].foreign_keys:
            parent_rows = rows.get(fk.ref_table) or []
            if not parent_rows:
                continue
            row = self._base_row(table)
            for child_col, parent_col in zip(fk.columns, fk.ref_columns):
                row[child_col] = parent_rows[0][parent_col]
            return row
        return None

    def _row_from_existing_child(self, table: str, rows: Rows) -> dict[str, Any] | None:
        for child_name, child_schema in self.registry.tables.items():
            child_rows = rows.get(child_name) or []
            if not child_rows:
                continue
            for fk in child_schema.foreign_keys:
                if fk.ref_table != table:
                    continue
                row = self._base_row(table)
                for child_col, parent_col in zip(fk.columns, fk.ref_columns):
                    row[parent_col] = child_rows[0][child_col]
                return row
        return None

    def _row_from_source_context(self, table: str, source_table: str, rows: Rows) -> dict[str, Any] | None:
        source_rows = rows.get(source_table) or []
        if not source_rows:
            return None
        row = self._base_row(table)
        source_row = source_rows[0]
        before = dict(row)
        self._link_child_to_parent(table, row, source_table, source_row)
        self._link_child_to_parent(source_table, source_row, table, row)
        return row if row != before else None

    def _apply_filters(self, rows: Rows, pair: TestPair) -> None:
        self._apply_filter_list(rows, pair, self.filters)

    def _satisfying_values_for_rows(
        self,
        table: str,
        column: str,
        filter_sig: FilterSignature,
        table_rows: list[dict[str, Any]],
        pair: TestPair,
    ) -> list[Any]:
        type_name = self.registry.tables[table].columns[column]
        source_ref = pair.measure.source_ref
        targets_measure_source = bool(
            source_ref
            and source_ref.casefold() == f"{table}.{column}".casefold()
        )
        if (
            pair.test_type == "ART"
            and targets_measure_source
            and self._numeric_type(type_name)
            and isinstance(filter_sig.value, (int, float))
        ):
            base = self._satisfying_value(type_name, filter_sig.operator, filter_sig.value)
            return [base + index for index, _ in enumerate(table_rows)]
        if (
            pair.test_type == "EST"
            and targets_measure_source
            and self._numeric_type(type_name)
            and isinstance(filter_sig.value, (int, float))
        ):
            base = self._satisfying_value(type_name, filter_sig.operator, filter_sig.value)
            split_count = int(pair.meta.get("split_count") or len(table_rows) or 1)
            if filter_sig.operator in {"<", "<=", "LT", "LTE"} and len(table_rows) > 1:
                return [base / len(table_rows) for _ in table_rows]
            if filter_sig.operator in {">", ">=", "GT", "GTE"} and len(table_rows) == 1:
                return [base * split_count]
            return [base for _ in table_rows]
        if (
            column in self.registry.tables[table].primary_key
            and self._date_like_type_or_value(type_name, filter_sig.value)
            and len(table_rows) > 1
        ):
            return [
                self._satisfying_date_value(filter_sig.operator, filter_sig.value, index=index)
                for index, _ in enumerate(table_rows, start=1)
            ]
        if len(table_rows) > 1 and self._column_participates_in_key(table, column):
            varied = self._distinct_satisfying_key_values(
                type_name,
                filter_sig.operator,
                filter_sig.value,
                len(table_rows),
            )
            if varied is not None:
                return varied
        value = self._satisfying_value(type_name, filter_sig.operator, filter_sig.value)
        return [value for _ in table_rows]

    def _column_participates_in_key(self, table: str, column: str) -> bool:
        schema = self.registry.tables[table]
        return any(column in key for key in [schema.primary_key, *schema.unique] if key)

    def _distinct_satisfying_key_values(
        self,
        type_name: str,
        operator: str,
        value: Any,
        count: int,
    ) -> list[Any] | None:
        if operator in {"=", "EQ", "IS NULL"}:
            return None
        base = self._satisfying_value(type_name, operator, value)
        direction = -1 if operator in {"<", "<=", "LT", "LTE"} else 1
        if isinstance(base, (int, float)):
            return [base + direction * index for index in range(count)]
        if isinstance(base, str) and base.isdigit():
            width = len(base)
            numeric = int(base)
            return [str(numeric + direction * index).zfill(width) for index in range(count)]
        return None

    def _filter_targets(self, source: str, rows: Rows) -> list[tuple[str, str]]:
        if "." in source:
            table, column = source.split(".", 1)
            if table in rows and table in self.registry.tables and column in self.registry.tables[table].columns:
                return [(table, column)]
            return []
        matches = [
            (table, source)
            for table in rows
            if table in self.registry.tables and source in self.registry.tables[table].columns
        ]
        return matches

    def _propagate_single_parent_keys(self, rows: Rows) -> None:
        for child_name, child_schema in self.registry.tables.items():
            child_rows = rows.get(child_name) or []
            if not child_rows:
                continue
            for fk in child_schema.foreign_keys:
                parent_rows = rows.get(fk.ref_table) or []
                if not parent_rows:
                    continue
                parent_keys = {
                    tuple(parent_row[parent_col] for parent_col in fk.ref_columns)
                    for parent_row in parent_rows
                }
                if len(parent_keys) != 1:
                    continue
                parent_key = next(iter(parent_keys))
                for child_row in child_rows:
                    for child_col, value in zip(fk.columns, parent_key):
                        child_row[child_col] = value

    def _dedupe_key_rows(self, rows: Rows) -> None:
        for table, table_rows in rows.items():
            keys = [
                key
                for key in [self.registry.tables[table].primary_key, *self.registry.tables[table].unique]
                if key
            ]
            if not keys:
                continue
            deduped: list[dict[str, Any]] = []
            seen_by_key = [set() for _ in keys]
            for row in table_rows:
                duplicate = False
                row_keys = []
                for key in keys:
                    value = tuple(row[col] for col in key)
                    row_keys.append(value)
                for seen, value in zip(seen_by_key, row_keys):
                    if value in seen:
                        duplicate = True
                        break
                if duplicate:
                    continue
                deduped.append(row)
                for seen, value in zip(seen_by_key, row_keys):
                    seen.add(value)
            rows[table] = deduped

    def _satisfying_value(self, type_name: str, operator: str, value: Any) -> Any:
        is_numeric = self._numeric_type(type_name)
        if operator == "IS NULL":
            return None
        if operator == "IS NOT NULL":
            return 1 if is_numeric else "value"
        if self._date_like_type_or_value(type_name, value):
            return self._satisfying_date_value(operator, value)
        if operator in {"=", "EQ"}:
            return value
        if operator in {"!=", "<>", "NEQ"}:
            if is_numeric and isinstance(value, (int, float)):
                return value + 1
            return f"{value}_other"
        if isinstance(value, (int, float)):
            if operator in {">", "GT"}:
                return value + 1
            if operator in {">=", "GTE"}:
                return value
            if operator in {"<", "LT"}:
                return value - 1
            if operator in {"<=", "LTE"}:
                return value
        return value

    def _non_satisfying_value(self, type_name: str, operator: str, value: Any) -> Any:
        is_numeric = self._numeric_type(type_name)
        if operator == "IS NULL":
            return 1 if is_numeric else "value"
        if operator == "IS NOT NULL":
            return None
        if self._date_like_type_or_value(type_name, value):
            return self._non_satisfying_date_value(operator, value)
        if operator in {"=", "EQ"}:
            if is_numeric and isinstance(value, (int, float)):
                return value + 1
            return f"{value}_other"
        if operator in {"!=", "<>", "NEQ"}:
            return value
        if isinstance(value, (int, float)):
            if operator in {">", ">=", "GT", "GTE"}:
                return value - 1
            if operator in {"<", "<=", "LT", "LTE"}:
                return value + 1
        return f"{value}_other" if isinstance(value, str) else value

    def _numeric_type(self, type_name: str) -> bool:
        return any(x in type_name.lower() for x in ("int", "decimal", "numeric", "real", "float", "double"))

    def _date_like_type_or_value(self, type_name: str, value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) >= 10
            and value[4:5] == "-"
            and value[7:8] == "-"
            and any(x in type_name.lower() for x in ("date", "time", "text"))
        )

    def _satisfying_date_value(self, operator: str, value: Any, index: int = 1) -> Any:
        if not isinstance(value, str) or len(value) < 10:
            return value
        try:
            year = int(value[:4])
        except ValueError:
            return value
        month = value[5:7] if value[5:7].isdigit() else "01"
        if operator in {"=", "EQ"}:
            return value
        day = min(max(index, 1), 28)
        if operator in {"<", "LT"}:
            return f"{year - 1:04d}-{month}-{day:02d}"
        if operator in {">", "GT"}:
            return f"{year + 1:04d}-{month}-{day:02d}"
        return f"{year:04d}-{month}-{day:02d}"

    def _non_satisfying_date_value(self, operator: str, value: Any) -> Any:
        if not isinstance(value, str) or len(value) < 10:
            return value
        try:
            year = int(value[:4])
        except ValueError:
            return value
        month = value[5:7] if value[5:7].isdigit() else "01"
        if operator in {"<", "<=", "LT", "LTE"}:
            return f"{year + 1:04d}-{month}-01"
        if operator in {">", ">=", "GT", "GTE"}:
            return f"{year - 1:04d}-{month}-01"
        if operator in {"=", "EQ"}:
            return f"{year + 1:04d}-{month}-01"
        return f"{year:04d}-{month}-01"
