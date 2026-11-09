from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .candidate_builder import build_candidate_space
from .contract import MetricContract
from .contract_checker import preserves_contract
from .contract_gate import validate_contract
from .evidence_scope import (
    SUPPORTED_POLICY,
    scoped_test_plans,
    witness_is_admissible,
)
from .pair_builder import PairBuilder
from .repair import repair_hint
from .schema_registry import SchemaRegistry
from .signature_extractor import extract_signature
from .static_rules import hard_static_grain_decision
from .validator import validate_pair


@dataclass
class PipelineResult:
    status: str
    results: list[dict[str, Any]]
    gate_errors: list[str]
    repair_hints: list[str]
    pass_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "results": self.results,
            "gate_errors": self.gate_errors,
            "repair_hints": self.repair_hints,
            "pass_class": self.pass_class,
        }


def run_pipeline(
    sql: str,
    registry: SchemaRegistry,
    contract: MetricContract,
    *,
    guard_static_pass_violations: bool = False,
    enable_static_alerts: bool = False,
    require_contract_preservation: bool = True,
    stabilize_signature: bool = True,
    evidence_scope_policy: str = SUPPORTED_POLICY,
) -> PipelineResult:
    """Run contract-relative static discharge and dynamic validation.

    ``guard_static_pass_violations`` is retained for call-site compatibility,
    but it no longer changes a dynamic decision. Absence of a static alert is
    not a proof of safety, so an executable contract-preserving violation must
    never be downgraded to PASS by the global static baseline.
    """
    candidates = build_candidate_space(registry)
    gate = validate_contract(contract, registry, candidates)
    if not gate.ok:
        return PipelineResult("UNSUPPORTED_CONTRACT", [], gate.errors, [])

    signature = extract_signature(sql)
    static_guard = None
    if enable_static_alerts:
        static_guard = hard_static_grain_decision(sql, registry, contract)
    pair_builder = PairBuilder(
        registry,
        contract,
        filters=signature.filters if stabilize_signature else [],
        conditional_filters=signature.conditional_filters if stabilize_signature else [],
        required_tables=signature.tables if stabilize_signature else set(),
        join_column_pairs=(
            signature.join_column_pairs if stabilize_signature else []
        ),
        group_columns=signature.groups if stabilize_signature else [],
        scalar_aggregate_comparisons=(
            signature.scalar_aggregate_comparisons if stabilize_signature else []
        ),
        projection_columns=(
            signature.direct_projections if stabilize_signature else []
        ),
        ranking_selection=(
            signature.has_order_limit if stabilize_signature else False
        ),
        ranking_count_sources=(
            [
                measure.source
                for measure in signature.measures
                if measure.alias == "ranking_metric"
                and measure.function == "COUNT"
            ]
            if stabilize_signature
            else []
        ),
    )
    results: list[dict[str, Any]] = []
    repair_hints: list[str] = []
    if enable_static_alerts and static_guard is not None and static_guard.status == "ALERT":
        results.append(
            {
                "status": "VIOLATION",
                "pass_class": None,
                "test_type": "STATIC_GRAIN_ALERT",
                "measure": None,
                "diagnosis": static_guard.reason,
            }
        )

    for measure in contract.measures:
        for scoped_plan in scoped_test_plans(
            measure,
            signature,
            registry,
            output_group_ref=contract.output_group.ref,
        ):
            plan = scoped_plan.plan
            if plan.static_status:
                results.append(
                    {
                        "status": plan.static_status,
                        "pass_class": plan.pass_class or "STATIC_PASS",
                        "test_type": plan.test_type,
                        "measure": measure.measure_id,
                        "diagnosis": plan.diagnosis or "supported static check passed",
                    }
                )
                continue
            try:
                pair = pair_builder.build(plan)
                pair_preserves_contract = preserves_contract(pair)
            except Exception as exc:
                results.append(
                    {
                        "status": "INVALID_WITNESS",
                        "pass_class": None,
                        "test_type": plan.test_type,
                        "measure": measure.measure_id,
                        "diagnosis": f"invalid witness: executable witness construction is not available for this shape ({exc})",
                    }
                )
                continue
            if require_contract_preservation and not pair_preserves_contract:
                results.append(
                    {
                        "status": "INVALID_WITNESS",
                        "pass_class": None,
                        "test_type": plan.test_type,
                        "measure": measure.measure_id,
                        "diagnosis": "invalid witness: constructed D0/D1 pair did not preserve contract value",
                    }
                )
                continue
            try:
                validation = validate_pair(
                    sql,
                    pair,
                    registry,
                    require_contract_preservation=require_contract_preservation,
                    contract_preserved=pair_preserves_contract,
                )
            except Exception as exc:
                results.append(
                    {
                        "status": "INVALID_WITNESS",
                        "pass_class": None,
                        "test_type": plan.test_type,
                        "measure": measure.measure_id,
                        "diagnosis": f"invalid witness: executable validation is not available for this shape ({exc})",
                    }
                )
                continue
            validation_result = validation.to_dict()
            validation_result["evidence_scope"] = scoped_plan.evidence_scope
            if scoped_plan.scope_reason:
                validation_result["scope_reason"] = scoped_plan.scope_reason
            if (
                validation_result["status"] == "VIOLATION"
                and not witness_is_admissible(
                    scoped_plan.evidence_scope,
                    evidence_scope_policy,
                )
            ):
                validation_result = {
                    **validation_result,
                    "status": "ABSTAIN",
                    "raw_status": "VIOLATION",
                    "pass_class": None,
                    "diagnosis": (
                        "deferred witness outside the supported evidence scope: "
                        f"{scoped_plan.scope_reason}"
                    ),
                }
            results.append(validation_result)
            if validation_result["status"] == "VIOLATION":
                repair_hints.append(repair_hint(sql, validation, measure, registry))

    if not results:
        return PipelineResult("UNSUPPORTED", [], [], [])
    if any(r["status"] == "VIOLATION" for r in results):
        return PipelineResult("VIOLATION", results, [], repair_hints)
    if any(r["status"] == "ABSTAIN" for r in results):
        return PipelineResult("ABSTAIN", results, [], [])
    if any(r["status"] == "INVALID_WITNESS" for r in results):
        return PipelineResult("INVALID_WITNESS", results, [], [])
    return PipelineResult("PASS", results, [], repair_hints, aggregate_pass_class(results))


def aggregate_pass_class(results: list[dict[str, Any]]) -> str:
    classes = [result.get("pass_class") for result in results if result.get("status") == "PASS"]
    if not classes:
        return "SCHEMA_PASS"
    if any(pass_class == "EXECUTABLE_PASS" for pass_class in classes):
        return "EXECUTABLE_PASS"
    if any(pass_class == "BENCHMARK_PROFILE_PASS" for pass_class in classes):
        return "BENCHMARK_PROFILE_PASS"
    if any(pass_class == "SCHEMA_PASS" for pass_class in classes):
        return "SCHEMA_PASS"
    return "EXECUTABLE_PASS"
