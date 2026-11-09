from __future__ import annotations

import json
from typing import Any

from .auto_repair import build_structured_repair_hint
from .contract import MetricContract
from .schema_registry import SchemaRegistry


def build_contract_ranking_prompt(
    row: dict[str, Any],
    registry: SchemaRegistry,
    contract_candidates: list[dict[str, Any]],
) -> str:
    scoped_tables = row.get("scoped_tables") or row.get("sql_tables") or []
    schema_text = render_schema(registry, scoped_tables)
    return (
        "You are selecting a metric-grain contract for analytical Text-to-SQL validation.\n"
        "The system has already enumerated all schema-valid contract candidates. "
        "You must not invent or edit contracts, tables, columns, grains, or semantics.\n"
        "Select exactly one candidate_id from the list when the question clearly implies it. "
        "If the correct contract is ambiguous or unsupported, return unsupported_reason.\n"
        "Return strict JSON only, with no markdown, no explanation, and no extra fields.\n\n"
        "Allowed output JSON shapes:\n"
        "{\n"
        '  "selected_candidate_id": "C0001"\n'
        "}\n"
        "or\n"
        "{\n"
        '  "unsupported_reason": "why no candidate should be selected"\n'
        "}\n\n"
        "Selection rules:\n"
        "- Choose only by candidate_id from Contract candidates.\n"
        "- Use the question and evidence to match the measure, aggregation word, entity/grain intent, and grouping phrase.\n"
        "- Prefer unsupported_reason when top candidates differ only by plausible business interpretation.\n"
        "- Do not use any gold SQL, candidate SQL, SQL signature, or execution result.\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Evidence:\n{row.get('evidence', '')}\n\n"
        f"Schema:\n{schema_text}\n\n"
        "Contract candidates:\n"
        f"{json.dumps(contract_candidates, ensure_ascii=False, indent=2)[:30000]}\n\n"
        "Selection JSON:"
    )


def build_factorized_contract_ranking_prompt(
    row: dict[str, Any],
    registry: SchemaRegistry,
    inventory: dict[str, Any],
) -> str:
    scoped_tables = row.get("scoped_tables") or row.get("sql_tables") or []
    schema_text = render_schema(registry, scoped_tables)
    strict = bool(
        (inventory.get("semantic_guidance") or {}).get("strict_semantic_rerun")
    )
    target_fields = (
        ',\n  "target_role": "DIRECT_ANSWER",\n'
        '  "target_source_quote": "exact substring from Question or Evidence"'
        if strict
        else ""
    )
    selection_precision_rule = (
        "- Precision is more important than coverage. Never select a merely topical or closest proxy. "
        "The frozen metric must be the requested answer metric or a clearly answer-determining "
        "internal component; otherwise return unsupported_reason.\n"
        "- target_role must be one of DIRECT_ANSWER, RANKING_METRIC, PREDICATE_METRIC, "
        "COMPARISON_LEFT, COMPARISON_RIGHT, RATIO_NUMERATOR, RATIO_DENOMINATOR, or "
        "DIFFERENCE_OPERAND. target_source_quote must be copied exactly from Question or "
        "Evidence and must support that role.\n"
        if strict
        else "- Prefer selecting the closest schema-grounded contract over unsupported_reason "
        "when the inventory contains atoms for the main counted entity or numeric metric.\n"
    )
    strict_event_rules = (
        "- For an event count, choose T_CGR on an identifier/FK atom from the event table. "
        "Its base_grain is the complete effective table grain shown in the atom, so repeated "
        "events are counted separately even when the chosen non-null source column is a grouping "
        "foreign key. Never use the parent/group key alone as the base grain.\n"
        "- Never select an atom whose base_grain_is_known_non_unique is true. That atom cannot "
        "represent one contribution per source row; return unsupported_reason if no safe atom "
        "remains.\n"
        "- Use PREDICATE_METRIC only when the selected frozen metric directly computes the "
        "aggregate tested by an explicit threshold. A distinct count of parent entities or a "
        "one-row-per-parent count is not an event-count predicate.\n"
        "- When the answer returns entities satisfying an event-count threshold (for example, "
        "teachers with at least two courses), use PREDICATE_METRIC and count course/event rows; "
        "do not use DIRECT_ANSWER or DET to count the teacher once.\n"
        "- When the answer returns the entity with most/fewest events but does not return the "
        "count itself, use RANKING_METRIC. When the question also explicitly asks for the count "
        "value, DIRECT_ANSWER may be appropriate.\n"
        "- If a direct question asks how many friends, visits, enrollments, or other event rows "
        "exist, do not use T_DET to count the parent person/entity; use T_CGR on the event table.\n"
        if strict
        else ""
    )
    component_rule = (
        "- For percentage/rate/ratio/difference metrics, select a component only when it is "
        "an exact named numerator, denominator, comparison side, or difference operand and label "
        "its exact component role. The component need not by itself reconstruct the final scalar, "
        "but it must be a necessary quantity whose remaining context can be held invariant by the "
        "candidate-specific alignment/evidence stage. Do not select a merely topical operand. "
        "Do not approximate "
        "MAX or MIN with SUM/AVG, do not approximate a ratio or difference with an unlabeled "
        "operand, and do not invent fields. Return unsupported_reason when the required formula "
        "is not represented.\n"
        if strict
        else "- For percentage/rate/ratio metrics derived from available numeric columns, choose "
        "the atom for the numerator or most semantically central component and select the closest "
        "SUM or AVG/TST template. Do not invent ratio fields.\n"
    )
    final_support_rule = (
        "- A display column is not automatically a safe grouping identity. Prefer the entity key "
        "for the operational output group when a non-unique name/title would merge entities.\n"
        "- In 'how many members' or similar possessive phrasing, count the named entities, not "
        "distinct values of their hometown, ZIP code, category, or other filtering attribute.\n"
        "- If the requested output group is a derived expression such as month extracted from a "
        "date, do not substitute the raw date column; return unsupported_reason when the inventory "
        "does not contain that derived group.\n"
        "- Return unsupported_reason when no listed atom/template is an exact direct metric or "
        "an exact, explicitly named diagnostic component with a declared target_role.\n"
        if strict
        else "- Return unsupported_reason only when no listed atom/template is relevant to the "
        "main metric or counted entity.\n"
    )
    return (
        "You are selecting a metric-grain contract for analytical Text-to-SQL validation.\n"
        "The system has already enumerated schema-grounded templates and typed atoms. "
        "You must not invent or edit templates, tables, columns, grains, semantics, or IDs.\n"
        "Select one template_id and the required atom IDs. "
        "If the correct contract is ambiguous or unsupported, return unsupported_reason.\n"
        "Return strict JSON only, with no markdown, no explanation, and no extra fields.\n\n"
        "Allowed output JSON shapes:\n"
        "{\n"
        '  "template_id": "T_CGR",\n'
        '  "measure_atom_id": null,\n'
        '  "entity_atom_id": "E0001",\n'
        '  "event_table_atom_id": null,\n'
        '  "output_group_atom_id": null'
        f"{target_fields}\n"
        "}\n"
        "or\n"
        "{\n"
        '  "unsupported_reason": "why no template/atom selection should be made"\n'
        "}\n\n"
        "Selection rules:\n"
        "- Choose IDs only from the inventory.\n"
        "- Satisfy the selected template's required_atoms exactly.\n"
        "- Use null for atom fields not required by the selected template.\n"
        "- Use null output_group_atom_id when the question does not ask for grouped output.\n"
        f"{selection_precision_rule}"
        "- SQL WHERE filters are outside this contract inventory. Do not return unsupported_reason only because a condition lacks a filter atom.\n"
        "- For questions that count entities satisfying filters, select the count contract for the counted entity grain and ignore filter columns unless they define the output metric itself.\n"
        "- Separate an entity's display attribute from its grouping identity. When the question asks for each/every entity and an entity key atom is available, use that key as output_group_atom_id; do not group only by a possibly non-unique name, title, or description column.\n"
        "- When the question asks how many categories such as continents, countries, languages, treatments, procedures, or types satisfy a condition, bind the counted entity atom to that category itself, not to a lower-level row that merely belongs to the category.\n"
        "- Words such as distinct kinds/types/categories imply a distinct category grain even when the SQL output later displays a human-readable category name.\n"
        "- When the answer is a categorical value such as a drug, diagnosis, procedure, label, or name, and the question filters that value by how many event rows occurred, select T_CGR for the event-row grain from the same table and use the categorical column as output_group_atom_id. Treat the count threshold or HAVING condition as a filter outside the contract.\n"
        "- Do not return unsupported_reason merely because the final answer is a categorical column; output_group_atom_id is exactly how categorical result values are represented.\n"
        "- When counting entities while another joined/event table supplies tests, scores, transactions, orders, activities, or other row-level conditions, prefer T_DET with the matching event_table_atom_id over T_CGR.\n"
        "- Do not select T_DET when the main metric is the number of event rows themselves, such as transactions, races, examinations, comments, votes, badges, posts, monthly records, or other repeated activities.\n"
        f"{strict_event_rules}"
        "- If the question says two or more / at least / HAVING COUNT(event) / number of event rows, distinguish the inner event-count condition from the outer metric. Use T_DET only for an outer count of unique entities; do not use it to replace the event count itself with COUNT(DISTINCT entity).\n"
        "- When the inventory semantic_guidance.event_count_semantics_detected flag is true, make this entity-count-vs-event-count decision explicitly before selecting T_DET; return unsupported_reason if the listed templates cannot represent the intended event-count metric.\n"
        "- Ranking metadata is added in Contract V2 independently of the legacy test family. "
        "For most/fewest/top/bottom by an event-row count, select T_CGR on a safe event-row atom "
        "with the requested entity/group as output_group_atom_id and use target_role=RANKING_METRIC. "
        "Do not abstain merely because T_TST_COUNT is absent.\n"
        "- When the answer lists entities selected by an aggregate event-count threshold, select "
        "the exact T_CGR event-row metric with target_role=PREDICATE_METRIC and the returned entity "
        "as output_group_atom_id. Do not abstain merely because the final projection is categorical.\n"
        "- For highest/top/most/max/min questions, prefer a TST template when a relevant numeric metric atom exists. Use the output_group atom for the requested ranked entity/group, or null if the question asks only for the scalar extreme value.\n"
        f"{component_rule}"
        f"{final_support_rule}"
        "- Do not use any gold SQL, candidate SQL, SQL signature, or execution result.\n\n"
        f"Question:\n{row['question']}\n\n"
        f"Evidence:\n{row.get('evidence', '')}\n\n"
        f"Schema:\n{schema_text}\n\n"
        "Schema-grounded factorized inventory:\n"
        f"{json.dumps(inventory, ensure_ascii=False, indent=2)[:30000]}\n\n"
        "Selection JSON:"
    )


def build_repair_prompt(
    row: dict[str, Any],
    candidate_sql: str,
    result: dict[str, Any],
    contract: MetricContract | None = None,
    registry: SchemaRegistry | None = None,
    structured_hint: dict[str, Any] | None = None,
) -> str | None:
    if result["status"] != "VIOLATION":
        return None
    if structured_hint is None and contract is not None:
        structured_hint = build_structured_repair_hint(contract, result, registry)
    hint_text = (
        json.dumps(structured_hint, ensure_ascii=False, indent=2)
        if structured_hint
        else json.dumps(result, ensure_ascii=False, indent=2)
    )
    schema_text = render_repair_schema(row, registry, contract, structured_hint)
    contract_text = render_contract_summary(contract)
    violation_text = render_violation_summary(result)
    repair_guide = render_type_specific_repair_guide(result, structured_hint)
    candidate_space = row.get("query_candidate_space") or row.get("db_candidate_space") or {}
    candidate_space_text = (
        json.dumps(candidate_space, ensure_ascii=False, indent=2)[:8000]
        if candidate_space
        else "Unavailable"
    )
    return (
        "You are repairing an analytical SQL query.\n"
        "Your job is to return a corrected SQLite query that passes the GrainTest witness below.\n"
        "The existing candidate may contain artificial fault-injection artifacts. "
        "Do not preserve comments, aliases, joins, columns, or expressions named grain_fault, __synthetic_fault__, "
        "__grain_fault_trigger__, or duplicate SUM/COUNT/AVG terms that exist only to trigger the witness.\n"
        "If the candidate is marked grain_fault:synthetic or is visibly missing question-required joins, filters, or formula components, "
        "reconstruct those required parts from the question, evidence, contract, and schema instead of preserving the incomplete synthetic SQL.\n"
        "Preserve the user's requested result shape: selected columns, grouping intent, ordering/top-k intent, and scalar-vs-row output. "
        "Preserve real filters from the question/evidence/candidate, but remove only the proven grain bug.\n\n"
        "What is wrong:\n"
        f"{violation_text}\n\n"
        "How to repair this error type:\n"
        f"{repair_guide}\n\n"
        f"Question:\n{row.get('question', '')}\n\n"
        f"Evidence:\n{row.get('evidence', '')}\n\n"
        "Relevant schema:\n"
        f"{schema_text}\n\n"
        "Contract grain semantics:\n"
        f"{contract_text}\n\n"
        "Schema-grounded candidate hints:\n"
        f"{candidate_space_text}\n\n"
        f"Candidate SQL:\n{candidate_sql}\n\n"
        "Structured repair hint:\n"
        f"{hint_text}\n\n"
        f"GrainTest witness:\n{json.dumps(result, ensure_ascii=False, indent=2)}\n\n"
        "Output constraints:\n"
        "- Return only one valid SQLite SQL query, with no markdown and no explanation.\n"
        "- Use only tables and columns from the schema.\n"
        "- Do not output any grain_fault or __synthetic_fault__ artifact.\n"
        "- The repaired query should be invariant under the witness: D0 and D1 should return the same value/top row."
    )


def render_repair_schema(
    row: dict[str, Any],
    registry: SchemaRegistry | None,
    contract: MetricContract | None,
    structured_hint: dict[str, Any] | None,
) -> str:
    if registry is None:
        return "Unavailable"
    tables: list[str] = []
    tables.extend(row.get("scoped_tables") or row.get("sql_tables") or [])
    tables.extend(contract_tables(contract))
    tables.extend(hint_tables(structured_hint))
    tables = ordered_unique(tables)
    if not tables:
        tables = list(registry.tables)
    return render_schema(registry, tables)


def contract_tables(contract: MetricContract | None) -> list[str]:
    if contract is None:
        return []
    tables = [contract.output_group.table]
    for measure in contract.measures:
        tables.append(measure.source_table)
        if measure.event_table:
            tables.append(measure.event_table)
        for ref in measure.base_grain:
            if "." in ref:
                tables.append(ref.split(".", 1)[0])
        if measure.entity_key and "." in measure.entity_key:
            tables.append(measure.entity_key.split(".", 1)[0])
    return tables


def hint_tables(structured_hint: dict[str, Any] | None) -> list[str]:
    if not structured_hint:
        return []
    tables: list[str] = []
    output_group = structured_hint.get("output_group")
    if isinstance(output_group, str) and "." in output_group:
        tables.append(output_group.split(".", 1)[0])
    for measure in structured_hint.get("measures", []) or []:
        source_table = measure.get("source_table")
        if source_table:
            tables.append(source_table)
        event_table = measure.get("event_table")
        if event_table:
            tables.append(event_table)
        for key in ("source_column", "entity_key"):
            value = measure.get(key)
            if isinstance(value, str) and "." in value:
                tables.append(value.split(".", 1)[0])
        for ref in measure.get("base_grain", []) or []:
            if isinstance(ref, str) and "." in ref:
                tables.append(ref.split(".", 1)[0])
    return tables


def render_contract_summary(contract: MetricContract | None) -> str:
    if contract is None:
        return "Unavailable"
    lines = [f"- output_group: {contract.output_group.ref}"]
    for measure in contract.measures:
        lines.extend(
            [
                f"- measure_id: {measure.measure_id}",
                f"  semantic_formula: {measure.semantic_formula}",
                f"  source: {measure.source_ref or measure.source_table}",
                f"  base_grain: {measure.base_grain}",
                f"  entity_key: {measure.entity_key}",
                f"  event_table: {measure.event_table}",
                f"  applicable_tests: {measure.applicable_tests}",
            ]
        )
    return "\n".join(lines)


def render_violation_summary(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for violation in result.get("results", []) or []:
        if violation.get("status") != "VIOLATION":
            continue
        lines.append(
            (
                f"- {violation.get('test_type')} failed for measure {violation.get('measure')}: "
                f"{violation.get('diagnosis') or 'grain invariant violation'}."
            )
        )
        lines.append(
            (
                "  Witness values: "
                f"contract D0={violation.get('contract_value_D0')}, "
                f"contract D1={violation.get('contract_value_D1')}, "
                f"SQL D0={violation.get('sql_output_D0')}, "
                f"SQL D1={violation.get('sql_output_D1')}."
            )
        )
    if not lines:
        return "- No per-measure violation details were provided."
    hints = result.get("repair_hints") or []
    for hint in hints:
        lines.append(f"- Existing repair hint: {hint}")
    return "\n".join(lines)


def render_type_specific_repair_guide(
    result: dict[str, Any],
    structured_hint: dict[str, Any] | None,
) -> str:
    test_types = sorted(
        {
            violation.get("test_type")
            for violation in result.get("results", []) or []
            if violation.get("status") == "VIOLATION" and violation.get("test_type")
        }
    )
    if not test_types and structured_hint:
        test_types = list(structured_hint.get("violation_test_types") or [])
    guides: list[str] = []
    for test_type in test_types:
        if test_type == "DET":
            guides.append(
                "- DET: The SQL is counting event/join rows instead of unique entities. "
                "Replace the affected COUNT with COUNT(DISTINCT entity key), or use an EXISTS semi-join so duplicate events cannot change the count."
            )
        elif test_type == "CGR":
            guides.append(
                "- CGR: The SQL counts a base-grain row after it has been replicated by a lower-grain join. "
                "Count COUNT(DISTINCT base_grain) or build a subquery with one row per base_grain before counting."
            )
        elif test_type in {"CRT", "GRT"}:
            guides.append(
                f"- {test_type}: The SQL sums a source measure after duplicated rows have been introduced. "
                "First construct a subquery with exactly one measure value per base_grain, then SUM that subquery. "
                "If the candidate has an added '+ SUM(CASE WHEN ...)' or synthetic trigger term, remove that duplicated term."
            )
        elif test_type == "ART":
            guides.append(
                "- ART: The SQL averages after non-uniform replication, so duplicated rows reweight the average. "
                "Average over a DISTINCT one-row-per-base_grain subquery that carries the source measure."
            )
        elif test_type == "EST":
            guides.append(
                "- EST: The SQL averages event values directly when the intended metric is entity-level. "
                "Aggregate events by entity first, then average the entity-level totals."
            )
        elif test_type == "TST":
            guides.append(
                "- TST: The top-k/ranking result changes when duplicated rows are injected. "
                "Compute the ranking metric once per output group/base_grain before ORDER BY/LIMIT. "
                "If the metric is a percentage or rate column already stored at that grain, do not SUM replicated copies; use the single value or an AVG over one row per grain. "
                "If the evidence defines a ratio, compute that ratio from the schema columns at the correct grain. "
                "Respect the question's direction: highest/most/max/top means ORDER BY metric DESC; lowest/least/min means ASC."
            )
    if not guides:
        guides.append("- Use the structured repair hint and witness values to remove only the grain-sensitive aggregation bug.")
    return "\n".join(guides)


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def render_schema(registry: SchemaRegistry, scoped_tables: list[str]) -> str:
    lines: list[str] = []
    for table_name in scoped_tables:
        if table_name not in registry.tables:
            continue
        table = registry.tables[table_name]
        cols = ", ".join(f"{name} {typ}" for name, typ in table.columns.items())
        lines.append(f"- {table_name}({cols})")
        if table.primary_key:
            lines.append(f"  primary_key: {table.primary_key}")
        if table.grain:
            lines.append(f"  grain: {table.grain}")
        for fk in table.foreign_keys:
            lines.append(f"  foreign_key: {fk.columns} -> {fk.ref_table}.{fk.ref_columns}")
    return "\n".join(lines)
