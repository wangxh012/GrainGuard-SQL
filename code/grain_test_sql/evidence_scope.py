from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .contract import MeasureContract
from .schema_registry import SchemaRegistry
from .signature_extractor import SQLSignature
from .template_matcher import TestPlan, match_tests


SUPPORTED = "SUPPORTED"
DEFERRED_TST_ONLY = "DEFERRED_TST_ONLY"
DEFERRED_INSTANCE_ONLY_UNIQUENESS = "DEFERRED_INSTANCE_ONLY_UNIQUENESS"

SUPPORTED_POLICY = "supported"
DEFER_TST_POLICY = "defer_tst_only"
UNRESTRICTED_POLICY = "unrestricted"
EVIDENCE_SCOPE_POLICIES = (
    SUPPORTED_POLICY,
    DEFER_TST_POLICY,
    UNRESTRICTED_POLICY,
)


@dataclass(frozen=True)
class ScopedTestPlan:
    """A verification plan whose witness eligibility is fixed before execution."""

    plan: TestPlan
    evidence_scope: str = SUPPORTED
    scope_reason: str | None = None


def scoped_test_plans(
    measure: MeasureContract,
    signature: SQLSignature,
    registry: SchemaRegistry,
    *,
    output_group_ref: str | None = None,
) -> list[ScopedTestPlan]:
    """Build plans and assign witness scope before any pair is executed.

    The declared-schema plan set is the executable search space.  A second,
    decision-only view enables already-recorded instance uniqueness facts.  If
    such a fact discharges a dynamic plan, a witness from that plan is retained
    as diagnostic evidence but cannot confirm a violation.  This comparison is
    candidate- and label-independent and happens before witness construction.
    """

    declared_registry = copy.deepcopy(registry)
    declared_registry.observed_uniqueness_policy = "provenance_only"
    declared_plans = match_tests(
        measure,
        signature,
        declared_registry,
        output_group_ref=output_group_ref,
    )

    profile_registry = copy.deepcopy(registry)
    profile_registry.observed_uniqueness_policy = (
        "conditional_pass"
        if registry.observed_uniqueness_scope_enabled
        else "provenance_only"
    )
    profile_plans = match_tests(
        measure,
        signature,
        profile_registry,
        output_group_ref=output_group_ref,
    )
    profile_dynamic = Counter(
        _plan_identity(plan) for plan in profile_plans if not plan.static_status
    )

    scoped: list[ScopedTestPlan] = []
    for plan in declared_plans:
        if plan.static_status:
            scoped.append(ScopedTestPlan(plan))
            continue
        if plan.test_type == "TST":
            scoped.append(
                ScopedTestPlan(
                    plan,
                    evidence_scope=DEFERRED_TST_ONLY,
                    scope_reason=(
                        "TST-only ranking evidence is exploratory and cannot "
                        "confirm a core violation"
                    ),
                )
            )
            continue

        identity = _plan_identity(plan)
        if profile_dynamic[identity]:
            profile_dynamic[identity] -= 1
            scoped.append(ScopedTestPlan(plan))
            continue
        scoped.append(
            ScopedTestPlan(
                plan,
                evidence_scope=DEFERRED_INSTANCE_ONLY_UNIQUENESS,
                scope_reason=(
                    "the plan is discharged only by undeclared uniqueness "
                    "observed in the candidate database instance"
                ),
            )
        )
    return scoped


def witness_is_admissible(evidence_scope: str, policy: str = SUPPORTED_POLICY) -> bool:
    """Return whether a witness can confirm VIOLATION under ``policy``."""

    if policy not in EVIDENCE_SCOPE_POLICIES:
        raise ValueError(f"unsupported evidence-scope policy: {policy}")
    if policy == UNRESTRICTED_POLICY:
        return True
    if policy == DEFER_TST_POLICY:
        return evidence_scope != DEFERRED_TST_ONLY
    return evidence_scope == SUPPORTED


def recorded_trace_status(
    row: dict[str, Any],
    policy: str = SUPPORTED_POLICY,
) -> str:
    """Apply the declared decision rule to a serialized exhaustive trace."""

    if row.get("unsupported_reason"):
        return "UNSUPPORTED"
    if row.get("static_violation"):
        return "VIOLATION"

    plans = row.get("plans") or []
    violating = [
        plan for plan in plans if plan.get("first_violation_rank") is not None
    ]
    if any(
        witness_is_admissible(
            str(plan.get("evidence_scope") or SUPPORTED),
            policy,
        )
        for plan in violating
    ):
        return "VIOLATION"
    if violating:
        return "ABSTAIN"
    if any(
        int(plan.get("candidate_count") or 0) == 0
        or not any(
            status in {"PASS", "VIOLATION"}
            for status in plan.get("execution_statuses") or []
        )
        for plan in plans
    ):
        return "INVALID_WITNESS"
    if row.get("static_pass") or plans:
        return "PASS"
    return "NO_VALID_EVIDENCE"


def emitted_trace_status(
    row: dict[str, Any],
    policy: str = SUPPORTED_POLICY,
) -> str:
    """Read a replay-emitted decision and verify it against the trace ledger."""

    computed = recorded_trace_status(row, policy)
    if row.get("evidence_scope_policy") != policy or not row.get("decision_status"):
        return computed
    emitted = str(row["decision_status"])
    if emitted != computed:
        raise ValueError(
            f"emitted decision {emitted} disagrees with scoped trace status {computed}"
        )
    return emitted


def _plan_identity(plan: TestPlan) -> tuple[object, ...]:
    return (
        plan.measure.measure_id,
        plan.test_type,
        plan.child_table,
        plan.replication_path,
        plan.entity_table,
    )
