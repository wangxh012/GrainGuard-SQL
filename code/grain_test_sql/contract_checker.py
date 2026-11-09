from __future__ import annotations

from statistics import mean
from typing import Any

from .pair_builder import Rows, TestPair


def contract_value(pair: TestPair, rows: Rows) -> Any:
    measure = pair.measure
    if measure.semantic_formula == "SUM_ONCE_PER_GRAIN":
        seen = set()
        total = 0
        for row in rows[measure.source_table]:
            key = tuple(row[g.split(".", 1)[1]] for g in measure.base_grain)
            if key in seen:
                continue
            seen.add(key)
            total += row[measure.source_column]
        return total

    if measure.semantic_formula == "AVG_ENTITY_TOTAL":
        if not measure.entity_key:
            raise ValueError("AVG_ENTITY_TOTAL requires entity_key")
        entity_col = measure.entity_key.split(".", 1)[1]
        event_entity_cols = pair.meta.get("event_entity_cols") or []
        totals: dict[Any, float] = {}
        for row in rows[measure.source_table]:
            if event_entity_cols:
                entity = tuple(row[col] for col in event_entity_cols)
            else:
                entity = row[entity_col]
            totals[entity] = totals.get(entity, 0) + row[measure.source_column]
        return mean(totals.values()) if totals else None

    if measure.semantic_formula == "COUNT_DISTINCT_ENTITY":
        if not measure.entity_key:
            raise ValueError("COUNT_DISTINCT_ENTITY requires entity_key")
        table, col = measure.entity_key.split(".", 1)
        event_table = (
            pair.meta.get("event_table")
            or measure.event_table
            or measure.source_table
        )
        values = set()
        if table != event_table and table in rows:
            values.update(row[col] for row in rows[table])
        else:
            event_entity_cols = pair.meta.get("event_entity_cols") or []
            for row in rows[event_table]:
                if event_entity_cols:
                    values.add(tuple(row[event_col] for event_col in event_entity_cols))
                elif col in row:
                    values.add(row[col])
        return len(values)

    if measure.semantic_formula == "COUNT_ONCE_PER_GRAIN":
        seen = set()
        for row in rows[measure.source_table]:
            key = tuple(row[g.split(".", 1)[1]] for g in measure.base_grain)
            seen.add(key)
        return len(seen)

    if measure.semantic_formula == "AVG_ONCE_PER_ENTITY":
        values = [row[measure.source_column] for row in rows[measure.source_table]]
        return mean(values) if values else None

    if measure.semantic_formula == "AVG_ONCE_PER_GRAIN":
        seen = set()
        values = []
        for row in rows[measure.source_table]:
            key = tuple(row[g.split(".", 1)[1]] for g in measure.base_grain)
            if key in seen:
                continue
            seen.add(key)
            values.append(row[measure.source_column])
        return mean(values) if values else None

    if measure.semantic_formula == "TOPK_BY_METRIC":
        return pair.meta.get("expected_top_group", pair.group_value)

    raise ValueError(f"Unsupported semantic formula: {measure.semantic_formula}")


def preserves_contract(pair: TestPair) -> bool:
    if contract_value(pair, pair.d0) != contract_value(pair, pair.d1):
        return False
    if pair.meta.get("preserve_output_group_domain"):
        table = pair.meta.get("entity_table") or pair.measure.source_table
        key_ref = pair.measure.entity_key
        if key_ref and "." in key_ref:
            key_column = key_ref.split(".", 1)[1]
        else:
            grain_refs = [
                ref
                for ref in pair.measure.base_grain
                if "." in ref and ref.split(".", 1)[0] == table
            ]
            key_column = (
                grain_refs[0].split(".", 1)[1]
                if grain_refs
                else None
            )
        if key_column:
            domain0 = {row.get(key_column) for row in pair.d0.get(table, [])}
            domain1 = {row.get(key_column) for row in pair.d1.get(table, [])}
            return domain0 == domain1
    return True
