from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema_registry import SchemaRegistry


MEASURE_NAME_HINTS = (
    "amount",
    "price",
    "revenue",
    "cost",
    "fee",
    "balance",
    "salary",
    "quantity",
    "qty",
    "total",
    "payment",
)
GROUP_NAME_HINTS = (
    "region",
    "city",
    "category",
    "status",
    "type",
    "channel",
    "level",
    "date",
    "month",
    "year",
)


@dataclass
class CandidateSpace:
    measure_candidates: list[dict[str, Any]]
    group_candidates: list[str]
    entity_candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure_candidates": self.measure_candidates,
            "group_candidates": self.group_candidates,
            "entity_candidates": self.entity_candidates,
        }


def is_numeric(type_name: str) -> bool:
    return any(x in type_name.lower() for x in ("int", "decimal", "numeric", "number", "real", "float", "double"))


def looks_like_code(column: str) -> bool:
    name = column.lower()
    return name.endswith("_id") or name in {"id", "code", "zip", "phone"}


def build_candidate_space(registry: SchemaRegistry) -> CandidateSpace:
    measures: list[dict[str, Any]] = []
    groups: list[str] = []
    entities: list[dict[str, Any]] = []

    for table_name, table in registry.tables.items():
        child_edges = registry.children_of(table_name)
        has_many_child = any(fk.many_allowed for _, fk in child_edges)
        is_leaf = not child_edges
        for column, type_name in table.columns.items():
            ref = f"{table_name}.{column}"
            is_pk = column in table.primary_key
            is_fk = any(column in fk.columns for fk in table.foreign_keys)
            lower = column.lower()

            if is_pk or is_fk or lower.endswith("_id"):
                entities.append({"column": ref, "possible_semantics": ["COUNT_DISTINCT_ENTITY"]})
                continue

            if is_numeric(type_name) and not looks_like_code(column):
                semantics: list[str]
                if "quantity" in lower or lower == "qty":
                    semantics = ["SUM", "SUM_ONCE_PER_GRAIN"]
                elif has_many_child:
                    semantics = ["SUM_ONCE_PER_GRAIN", "ENTITY_TOTAL_INPUT", "AVG_ONCE_PER_GRAIN"]
                elif is_leaf:
                    semantics = ["SUM", "SUM_ONCE_PER_GRAIN", "AVG_EVENT_VALUE", "AVG_ONCE_PER_GRAIN"]
                else:
                    semantics = ["SUM_ONCE_PER_GRAIN", "AVG_ONCE_PER_ENTITY"]
                measures.append(
                    {
                        "column": ref,
                        "table_grain": registry.grain_refs(table_name),
                        "possible_semantics": semantics,
                    }
                )
                continue

            if not is_numeric(type_name) or any(h in lower for h in GROUP_NAME_HINTS):
                groups.append(ref)

    return CandidateSpace(measures, sorted(set(groups)), entities)
