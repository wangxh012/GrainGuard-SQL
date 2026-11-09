from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldSemanticAuditDecision:
    status: str
    rule_id: str
    reason: str


def audit_gold_sql_semantics(
    question: str,
    sql: str,
    *,
    current_status: str | None = None,
) -> GoldSemanticAuditDecision | None:
    """Question-aware audit for GOLD SQL dry-runs.

    These rules intentionally stay outside the default pipeline. They use the
    natural-language question to separate intrinsic GOLD errors from contract
    boundary triggers, and should only be enabled for reference-SQL auditing.
    """
    q = _normalize(question)
    s = _normalize(sql)

    violation = _confirmed_violation(q, s)
    if violation:
        return violation
    if current_status == "VIOLATION":
        return _contract_boundary_pass(q, s)
    return None


def apply_gold_semantic_audit(
    result: dict,
    decision: GoldSemanticAuditDecision,
) -> dict:
    if decision.status == "VIOLATION":
        return {
            **result,
            "status": "VIOLATION",
            "pass_class": None,
            "results": [
                *result.get("results", []),
                _audit_result(decision, original_status=result.get("status")),
            ],
            "gold_semantic_audit": decision.__dict__,
        }

    if decision.status == "PASS" and result.get("status") == "VIOLATION":
        audited_results = []
        for row in result.get("results", []):
            if row.get("status") == "VIOLATION":
                audited_results.append(
                    {
                        **row,
                        "status": "PASS",
                        "pass_class": "QUESTION_AUDITED_PASS",
                        "guarded_violation_status": "VIOLATION",
                        "question_audit_rule": decision.rule_id,
                        "question_audit_reason": decision.reason,
                    }
                )
            else:
                audited_results.append(row)
        audited_results.append(_audit_result(decision, original_status=result.get("status")))
        return {
            **result,
            "status": "PASS",
            "pass_class": "QUESTION_AUDITED_PASS",
            "results": audited_results,
            "gold_semantic_audit": decision.__dict__,
        }

    return result


def _confirmed_violation(q: str, s: str) -> GoldSemanticAuditDecision | None:
    if "in one semester" in q and "student_enrolment" in s and "semester_id" not in s:
        return GoldSemanticAuditDecision(
            "VIOLATION",
            "missing_required_semester_group",
            "Question requires two degree programs in one semester, but SQL groups only by student and omits semester_id.",
        )

    if _asks_scalar_count(q) and _selects_grouped_count_only(s):
        return GoldSemanticAuditDecision(
            "VIOLATION",
            "scalar_count_returned_per_group",
            "Question asks for a scalar count, but SQL returns one count per GROUP BY group.",
        )

    if _asks_predominant_language(q) and _filters_spanish_without_country_max(s):
        return GoldSemanticAuditDecision(
            "VIOLATION",
            "missing_per_country_language_max",
            "Question requires Spanish to be the largest-percentage language per country, but SQL only filters Spanish rows.",
        )

    if _asks_money_spent_ranking(q) and _orders_by_count(s):
        return GoldSemanticAuditDecision(
            "VIOLATION",
            "money_ranking_uses_count",
            "Question ranks by money spent, but SQL orders by treatment count instead of SUM(cost).",
        )

    if _asks_distinct_types(q) and _having_count_without_distinct(s):
        return GoldSemanticAuditDecision(
            "VIOLATION",
            "type_count_without_distinct",
            "Question asks for distinct types, but SQL counts rows without COUNT(DISTINCT type).",
        )

    return None


def _contract_boundary_pass(q: str, s: str) -> GoldSemanticAuditDecision | None:
    if _grouped_extreme_aggregate(q, s):
        return GoldSemanticAuditDecision(
            "PASS",
            "grouped_extreme_aggregate",
            "Question asks for a grouped MIN/MAX/AVG aggregate; witness mutation is a contract boundary trigger.",
        )

    if _asks_event_count_threshold(q) and _having_count_without_distinct(s):
        return GoldSemanticAuditDecision(
            "PASS",
            "event_count_threshold",
            "Question asks for an event-count threshold, so HAVING COUNT(*) is aligned with the question.",
        )

    if _asks_event_count_ranking(q) and _orders_by_count(s):
        return GoldSemanticAuditDecision(
            "PASS",
            "event_count_ranking",
            "Question ranks by number of events, so ORDER BY COUNT(*) is aligned with the question.",
        )

    if _asks_money_spent_ranking(q) and _orders_by_sum(s):
        return GoldSemanticAuditDecision(
            "PASS",
            "money_ranking_uses_sum",
            "Question ranks by total money and SQL orders by SUM(cost).",
        )

    if "car maker" in q and " in each continent" in q and "car_makers" in s and "count(*)" in s:
        return GoldSemanticAuditDecision(
            "PASS",
            "car_maker_event_count_by_continent",
            "Question asks for car-maker counts by continent; counting car_makers rows is the intended measure.",
        )

    if "treatment dates" in q and "date_of_treatment" in s:
        return GoldSemanticAuditDecision(
            "PASS",
            "detail_rows_expected",
            "Question asks for treatment-date detail rows, so repeated dog rows across dates are expected.",
        )

    return None


def _audit_result(decision: GoldSemanticAuditDecision, *, original_status: str | None) -> dict:
    return {
        "status": decision.status,
        "pass_class": "QUESTION_AUDITED_PASS" if decision.status == "PASS" else None,
        "test_type": "GOLD_SEMANTIC_AUDIT",
        "measure": "question_sql_semantics",
        "diagnosis": decision.reason,
        "audit_rule": decision.rule_id,
        "original_status": original_status,
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _asks_scalar_count(q: str) -> bool:
    if any(token in q for token in (" for each ", " each ", " per ", " by ")):
        return False
    if q.startswith(("which ", "what are ", "show ", "list ", "return the codes")):
        return False
    return (
        q.startswith("how many ")
        or q.startswith("count ")
        or "what is the number of" in q
        or "total number of" in q
    )


def _selects_grouped_count_only(s: str) -> bool:
    if " group by " not in s or " order by " in s:
        return False
    select_part = s.split(" from ", 1)[0]
    return bool(re.fullmatch(r"select\s+count\s*\(\s*(\*|[^)]+)\s*\)\s*", select_part))


def _asks_predominant_language(q: str) -> bool:
    return "spanish" in q and ("largest percentage" in q or "predominant" in q)


def _filters_spanish_without_country_max(s: str) -> bool:
    if "countrylanguage" not in s or "spanish" not in s or " group by countrycode" not in s:
        return False
    return s.count("select ") == 1


def _asks_money_spent_ranking(q: str) -> bool:
    return any(
        phrase in q
        for phrase in (
            "spent the most",
            "most money",
            "largest amount of money",
            "spent the largest amount",
        )
    )


def _orders_by_count(s: str) -> bool:
    return bool(re.search(r"order\s+by\s+count\s*\(", s))


def _orders_by_sum(s: str) -> bool:
    return bool(re.search(r"order\s+by\s+sum\s*\(", s))


def _asks_distinct_types(q: str) -> bool:
    return any(
        phrase in q
        for phrase in (
            "types of",
            "type of",
            "different types",
            "two or more types",
            "at least two types",
        )
    )


def _having_count_without_distinct(s: str) -> bool:
    return " having " in s and bool(re.search(r"count\s*\(\s*\*\s*\)", s)) and "distinct" not in s


def _grouped_extreme_aggregate(q: str, s: str) -> bool:
    if " group by " not in s or " order by " in s:
        return False
    if not any(func in s for func in ("max(", "min(", "avg(")):
        return False
    return any(word in q for word in ("maximum", "minimum", "average", "max ", "min "))


def _asks_event_count_threshold(q: str) -> bool:
    if _asks_distinct_types(q) or "in one semester" in q:
        return False
    event_words = (
        "pet",
        "course",
        "vote",
        "friend",
        "treatment",
        "visit",
        "orchestra",
        "like",
    )
    threshold_words = (
        "more than one",
        "at least two",
        "two or more",
        "more than once",
        "at least 3",
        "3 or more",
    )
    return any(word in q for word in event_words) and any(phrase in q for phrase in threshold_words)


def _asks_event_count_ranking(q: str) -> bool:
    if _asks_money_spent_ranking(q):
        return False
    return (
        "most number of" in q
        or "fewest number of" in q
        or "paid for the most treatments" in q
        or ("most" in q and any(word in q for word in ("flights", "treatments", "concerts")))
    )
