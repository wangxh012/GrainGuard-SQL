from __future__ import annotations

from dataclasses import dataclass

from .candidate_builder import CandidateSpace
from .contract import MetricContract, MeasureContract
from .schema_registry import SchemaRegistry


COMPLEX_CONDITION_KINDS = {
    "BOOLEAN_OR",
    "BOOLEAN_NOT",
    "DATE_EXPRESSION",
    "AGGREGATE_COMPARISON",
    "EXISTS",
    "NOT_EXISTS",
    "RELATIONAL_PREDICATE",
    "TYPE_NORMALIZATION",
}


@dataclass(frozen=True)
class GateResult:
    ok: bool
    errors: list[str]


def validate_contract(
    contract: MetricContract,
    registry: SchemaRegistry,
    candidates: CandidateSpace,
    *,
    allow_complex_conditions: bool = False,
) -> GateResult:
    errors: list[str] = []
    measure_by_col = {m["column"]: m for m in candidates.measure_candidates}
    entity_cols = {e["column"] for e in candidates.entity_candidates}
    group_cols = set(candidates.group_candidates)

    if contract.output_group.ref not in group_cols and not registry.column_exists(contract.output_group.ref):
        errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FIELD output_group={contract.output_group.ref}")

    errors.extend(
        _validate_v2_contract_fields(
            contract,
            registry,
            allow_complex_conditions=allow_complex_conditions,
        )
    )

    for measure in contract.measures:
        errors.extend(_validate_measure(measure, contract.output_group.ref, registry, measure_by_col, entity_cols))

    return GateResult(ok=not errors, errors=errors)


def _validate_measure(
    measure: MeasureContract,
    output_group: str,
    registry: SchemaRegistry,
    measure_by_col: dict[str, dict],
    entity_cols: set[str],
) -> list[str]:
    errors: list[str] = []

    if measure.semantic_formula == "TOPK_BY_METRIC":
        if measure.source_ref and not registry.column_exists(measure.source_ref):
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FIELD source_column={measure.source_ref}")
            return errors
        allowed = {"TOPK_BY_METRIC"}
    elif measure.semantic_formula == "COUNT_DISTINCT_ENTITY":
        if not measure.entity_key or (
            measure.entity_key not in entity_cols and not registry.column_exists(measure.entity_key)
        ):
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FIELD entity_key={measure.entity_key}")
    else:
        if measure.semantic_formula == "COUNT_ONCE_PER_GRAIN":
            if not measure.source_ref or not registry.column_exists(measure.source_ref):
                errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FIELD source_column={measure.source_ref}")
                return errors
            allowed = {"COUNT_ONCE_PER_GRAIN"}
        elif not measure.source_ref or measure.source_ref not in measure_by_col:
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FIELD source_column={measure.source_ref}")
            return errors
        else:
            allowed = set(measure_by_col[measure.source_ref]["possible_semantics"])
        if measure.semantic_formula == "AVG_ENTITY_TOTAL":
            pass
        elif measure.semantic_formula not in allowed:
            if measure.semantic_formula != "COUNT_ONCE_PER_GRAIN":
                errors.append(f"UNSUPPORTED_CONTRACT_INVALID_SEMANTIC {measure.source_ref}:{measure.semantic_formula}")

    for grain_ref in measure.base_grain:
        if not registry.column_exists(grain_ref):
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_GRAIN {grain_ref}")

    counting_object = measure.counting_object or {}
    counting_ref = counting_object.get("ref")
    if counting_ref and not registry.column_exists(str(counting_ref)):
        errors.append(f"UNSUPPORTED_CONTRACT_INVALID_COUNTING_OBJECT {counting_ref}")

    return errors


def _validate_v2_contract_fields(
    contract: MetricContract,
    registry: SchemaRegistry,
    *,
    allow_complex_conditions: bool = False,
) -> list[str]:
    if contract.contract_version < 2:
        return []
    errors: list[str] = []
    allowed_roles = {
        "DIRECT_ANSWER",
        "RANKING_METRIC",
        "PREDICATE_METRIC",
        "COMPARISON_LEFT",
        "COMPARISON_RIGHT",
        "RATIO_NUMERATOR",
        "RATIO_DENOMINATOR",
        "DIFFERENCE_OPERAND",
    }
    role = str((contract.target or {}).get("role") or "")
    if role not in allowed_roles:
        errors.append(f"UNSUPPORTED_CONTRACT_INVALID_TARGET_ROLE {role}")
    if not str((contract.target or {}).get("source_quote") or "").strip():
        errors.append("UNSUPPORTED_CONTRACT_MISSING_TARGET_SOURCE_QUOTE")
    if contract.filter_logic != "AND":
        errors.append(f"UNSUPPORTED_CONTRACT_FILTER_LOGIC {contract.filter_logic}")
    if contract.complex_conditions and not allow_complex_conditions:
        # Diagnostic V2.1 bindings are not yet executed by PairBuilder or by
        # contract preservation. Keep them outside the downstream-supported
        # cohort until both paths consume the same condition representation.
        errors.append("UNSUPPORTED_CONTRACT_NONEXECUTABLE_COMPLEX_CONDITIONS")
    mention_by_id: dict[str, str] = {}
    for index, item in enumerate(contract.filter_mentions):
        mention_id = str(item.get("mention_id") or "")
        kind = str(item.get("kind") or "")
        if not mention_id or mention_id in mention_by_id:
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_MENTION_ID {index}:{mention_id}")
        if kind not in {
            "ROW_FILTER",
            "AGGREGATE_PREDICATE",
            "RANKING_CONTEXT",
            "COMPARISON_CONTEXT",
            "ANSWER_CONTEXT",
        }:
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_MENTION_KIND {index}:{kind}")
        mention_by_id[mention_id] = kind
    mapped_mentions: dict[str, int] = {}
    allowed_operators = {
        "=", "!=", ">", ">=", "<", "<=", "BETWEEN", "IN", "LIKE",
        "IS NULL", "IS NOT NULL",
    }
    for index, item in enumerate(contract.filters):
        mention_id = str(item.get("mention_id") or "")
        operator = str(item.get("operator") or "").upper()
        scope = str(item.get("scope") or "").upper()
        if mention_by_id.get(mention_id) != "ROW_FILTER":
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_MENTION {index}:{mention_id}")
        else:
            mapped_mentions[mention_id] = mapped_mentions.get(mention_id, 0) + 1
        errors.extend(_validate_filter_operand(index, item, registry))
        if operator not in allowed_operators:
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_OPERATOR {index}:{operator}")
        if scope not in {"METRIC_INPUT", "ANSWER_CONTEXT"}:
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_SCOPE {index}:{scope}")
    for index, item in enumerate(contract.complex_conditions):
        mention_id = str(item.get("mention_id") or "")
        kind = str(item.get("kind") or "").upper()
        scope = str(item.get("scope") or "").upper()
        columns = item.get("columns")
        if mention_by_id.get(mention_id) != "ROW_FILTER":
            errors.append(
                f"UNSUPPORTED_CONTRACT_INVALID_COMPLEX_CONDITION_MENTION {index}:{mention_id}"
            )
        else:
            mapped_mentions[mention_id] = mapped_mentions.get(mention_id, 0) + 1
        if kind not in COMPLEX_CONDITION_KINDS:
            errors.append(
                f"UNSUPPORTED_CONTRACT_INVALID_COMPLEX_CONDITION_KIND {index}:{kind}"
            )
        if scope not in {"METRIC_INPUT", "ANSWER_CONTEXT"}:
            errors.append(
                f"UNSUPPORTED_CONTRACT_INVALID_COMPLEX_CONDITION_SCOPE {index}:{scope}"
            )
        if not isinstance(columns, list) or not columns:
            errors.append(
                f"UNSUPPORTED_CONTRACT_INVALID_COMPLEX_CONDITION_COLUMNS {index}"
            )
            columns = []
        for column in columns:
            if not registry.column_exists(str(column or "")):
                errors.append(
                    f"UNSUPPORTED_CONTRACT_INVALID_COMPLEX_CONDITION_COLUMN "
                    f"{index}:{column}"
                )
        if not str(item.get("source_quote") or "").strip():
            errors.append(
                f"UNSUPPORTED_CONTRACT_MISSING_COMPLEX_CONDITION_SOURCE_QUOTE {index}"
            )
    for mention_id, kind in mention_by_id.items():
        if kind == "ROW_FILTER" and mapped_mentions.get(mention_id) != 1:
            errors.append(f"UNSUPPORTED_CONTRACT_UNMAPPED_FILTER_MENTION {mention_id}")
    for group in contract.output_groups:
        if not registry.column_exists(group.ref):
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_OUTPUT_GROUP {group.ref}")
    output_semantics = contract.output_group_semantics or {}
    output_mode = str(output_semantics.get("mode") or "")
    if output_mode not in {"PRESERVE", "NONE", "REPLACE"}:
        errors.append(f"UNSUPPORTED_CONTRACT_INVALID_OUTPUT_GROUP_MODE {output_mode}")
    if output_mode == "NONE" and contract.output_groups:
        errors.append("UNSUPPORTED_CONTRACT_NONE_WITH_OUTPUT_GROUPS")
    if output_mode == "REPLACE" and not contract.output_groups:
        errors.append("UNSUPPORTED_CONTRACT_REPLACE_WITHOUT_OUTPUT_GROUPS")
    if contract.ranking is not None:
        k = contract.ranking.get("k")
        direction = str(contract.ranking.get("direction") or "")
        tie_policy = str(contract.ranking.get("tie_policy") or "")
        if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_RANKING_K {k}")
        if direction not in {"ASC", "DESC", "UNSPECIFIED"}:
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_RANKING_DIRECTION {direction}")
        if tie_policy not in {"EXACT_K", "WITH_TIES", "UNIQUE_EXTREMUM", "UNSPECIFIED"}:
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_TIE_POLICY {tie_policy}")
    return errors


def _validate_filter_operand(
    index: int,
    item: dict,
    registry: SchemaRegistry,
) -> list[str]:
    errors: list[str] = []
    raw_operand = item.get("left_operand")
    if raw_operand is None:
        raw_operand = {
            "kind": "COLUMN",
            "columns": [item.get("column")],
            "scale": 1,
        }
    if not isinstance(raw_operand, dict):
        return [f"UNSUPPORTED_CONTRACT_INVALID_FILTER_OPERAND {index}"]
    kind = str(raw_operand.get("kind") or "").upper()
    columns = raw_operand.get("columns")
    if kind not in {"COLUMN", "SUM", "RATIO", "DIFFERENCE", "PRODUCT"}:
        errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_OPERAND_KIND {index}:{kind}")
    if not isinstance(columns, list):
        errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_OPERAND_COLUMNS {index}")
        columns = []
    expected = 1 if kind == "COLUMN" else 2 if kind in {"RATIO", "DIFFERENCE"} else None
    if expected is not None and len(columns) != expected:
        errors.append(
            f"UNSUPPORTED_CONTRACT_INVALID_FILTER_OPERAND_ARITY {index}:{kind}:{len(columns)}"
        )
    if kind in {"SUM", "PRODUCT"} and len(columns) < 2:
        errors.append(
            f"UNSUPPORTED_CONTRACT_INVALID_FILTER_OPERAND_ARITY {index}:{kind}:{len(columns)}"
        )
    for column in columns:
        if not registry.column_exists(str(column or "")):
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_COLUMN {index}:{column}")
    scale = raw_operand.get("scale", 1)
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale == 0:
        errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_OPERAND_SCALE {index}:{scale}")
    provenance = item.get("value_provenance")
    if provenance is not None:
        if not isinstance(provenance, dict) or provenance.get("kind") not in {
            "QUESTION_OR_EVIDENCE_LITERAL",
            "LINKED_SCHEMA_SAMPLE",
            "NORMALIZED_FROM_GROUNDED_TEXT",
            "NULL_OPERATOR",
        }:
            errors.append(f"UNSUPPORTED_CONTRACT_INVALID_FILTER_VALUE_PROVENANCE {index}")
    return errors
