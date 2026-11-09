from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OutputGroup:
    table: str
    column: str

    @property
    def ref(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass(frozen=True)
class MeasureContract:
    measure_id: str
    source_table: str
    source_column: str | None
    base_grain: list[str]
    semantic_formula: str
    applicable_tests: list[str]
    entity_key: str | None = None
    event_table: str | None = None
    ranking_metric_semantic: str | None = None
    counting_object: dict[str, Any] | None = None

    @property
    def source_ref(self) -> str | None:
        if not self.source_column:
            return None
        return f"{self.source_table}.{self.source_column}"


@dataclass(frozen=True)
class MetricContract:
    output_group: OutputGroup
    measures: list[MeasureContract]
    contract_version: int = 1
    target: dict[str, Any] | None = None
    filter_mentions: list[dict[str, Any]] = field(default_factory=list)
    filters: list[dict[str, Any]] = field(default_factory=list)
    filter_logic: str = "AND"
    ranking: dict[str, Any] | None = None
    output_groups: list[OutputGroup] = field(default_factory=list)
    output_group_semantics: dict[str, Any] | None = None
    contract_revision: str | None = None
    complex_conditions: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> "MetricContract":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetricContract":
        group = raw["output_group"]
        measures = []
        for m in raw["measures"]:
            measures.append(
                MeasureContract(
                    measure_id=m["measure_id"],
                    source_table=m["source_table"],
                    source_column=m.get("source_column"),
                    base_grain=list(m.get("base_grain", [])),
                    semantic_formula=m["semantic_formula"],
                    applicable_tests=list(m.get("applicable_tests", [])),
                    entity_key=m.get("entity_key"),
                    event_table=m.get("event_table"),
                    ranking_metric_semantic=m.get("ranking_metric_semantic"),
                    counting_object=(
                        dict(m["counting_object"])
                        if isinstance(m.get("counting_object"), dict)
                        else None
                    ),
                )
            )
        output_groups = [
            OutputGroup(item["table"], item["column"])
            for item in raw.get("output_groups", [])
            if isinstance(item, dict) and item.get("table") and item.get("column")
        ]
        if "output_groups" not in raw:
            output_groups = [OutputGroup(group["table"], group["column"])]
        return cls(
            output_group=OutputGroup(group["table"], group["column"]),
            measures=measures,
            contract_version=int(raw.get("contract_version", 1)),
            target=(dict(raw["target"]) if isinstance(raw.get("target"), dict) else None),
            filter_mentions=[
                dict(item)
                for item in raw.get("filter_mentions", [])
                if isinstance(item, dict)
            ],
            filters=[dict(item) for item in raw.get("filters", []) if isinstance(item, dict)],
            filter_logic=str(raw.get("filter_logic") or "AND"),
            ranking=(dict(raw["ranking"]) if isinstance(raw.get("ranking"), dict) else None),
            output_groups=output_groups,
            output_group_semantics=(
                dict(raw["output_group_semantics"])
                if isinstance(raw.get("output_group_semantics"), dict)
                else None
            ),
            contract_revision=(
                str(raw["contract_revision"])
                if raw.get("contract_revision") is not None
                else None
            ),
            complex_conditions=[
                dict(item)
                for item in raw.get("complex_conditions", [])
                if isinstance(item, dict)
            ],
        )
