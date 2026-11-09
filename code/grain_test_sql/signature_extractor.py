from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class MeasureSignature:
    alias: str
    function: str
    source: str | None
    distinct: bool = False
    row_counting: bool = False


@dataclass(frozen=True)
class FilterSignature:
    source: str
    operator: str
    value: object


@dataclass(frozen=True)
class ScalarAggregateComparison:
    """Outer-row predicate compared with a scalar aggregate subquery."""

    source: str
    operator: str
    aggregate_function: str
    aggregate_source: str | None


@dataclass
class SQLSignature:
    sql: str = ""
    tables: set[str] = field(default_factory=set)
    # Tables participating in the outer SELECT only. ``tables`` deliberately
    # remains the transitive set because witness execution must materialize
    # relations referenced by scalar/IN subqueries as well.
    outer_tables: set[str] = field(default_factory=set)
    joins: list[tuple[str, str]] = field(default_factory=list)
    join_column_pairs: list[tuple[str, str]] = field(default_factory=list)
    measures: list[MeasureSignature] = field(default_factory=list)
    projections: list[str] = field(default_factory=list)
    # Only top-level column projections, excluding columns nested inside
    # arithmetic expressions or aggregate calls.
    direct_projections: list[str] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    order_columns: list[str] = field(default_factory=list)
    filters: list[FilterSignature] = field(default_factory=list)
    conditional_filters: list[FilterSignature] = field(default_factory=list)
    scalar_aggregate_comparisons: list[ScalarAggregateComparison] = field(
        default_factory=list
    )
    distinct: bool = False
    has_order_limit: bool = False
    has_subquery: bool = False
    synthetic_fault: bool = False


def extract_signature(sql: str, dialect: str = "sqlite") -> SQLSignature:
    tree = sqlglot.parse_one(sql, read=dialect)
    sig = SQLSignature(sql=sql, synthetic_fault="grain_fault:synthetic" in sql)

    aliases: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        name = table.name
        alias = table.alias_or_name
        aliases[alias] = name
        aliases[name] = name
        sig.tables.add(name)

    select = _root_select(tree)
    if select is None:
        return sig

    outer_aliases = _aliases_for_select(select)
    sig.outer_tables = set(outer_aliases.values())
    sig.has_subquery = any(True for _ in select.find_all(exp.Subquery))

    for join in select.find_all(exp.Join):
        if _nearest_select(join) is not select:
            continue
        joined = join.this
        if isinstance(joined, exp.Table):
            # Store relation to any table referenced in the ON clause; this is a lightweight signature.
            on_tables = _tables_in_expr(join.args.get("on"), outer_aliases)
            for left in on_tables:
                if left != outer_aliases.get(joined.alias_or_name, joined.name):
                    sig.joins.append(
                        (
                            left,
                            outer_aliases.get(joined.alias_or_name, joined.name),
                        )
                    )
            on = join.args.get("on")
            if on is not None:
                for equality in on.find_all(exp.EQ):
                    if not isinstance(equality.left, exp.Column) or not isinstance(
                        equality.right,
                        exp.Column,
                    ):
                        continue
                    left_ref = _column_ref(equality.left, outer_aliases)
                    right_ref = _column_ref(equality.right, outer_aliases)
                    if left_ref and right_ref:
                        sig.join_column_pairs.append((left_ref, right_ref))

    sig.distinct = bool(select.args.get("distinct"))
    for projection in select.expressions:
        ref = _column_ref(projection, outer_aliases)
        if ref:
            sig.projections.append(ref)
            sig.direct_projections.append(ref)
        else:
            sig.projections.extend(
                _unique_column_refs_in_select(projection, outer_aliases, select)
            )
        for agg in _aggregates_in_select(projection, select):
            alias = projection.alias_or_name or agg.sql(dialect="sqlite")
            source = _first_column_ref(agg, outer_aliases)
            sig.measures.append(
                MeasureSignature(
                    alias=alias,
                    function=agg.key.upper(),
                    source=source,
                    distinct=_is_distinct_agg(agg),
                    row_counting=_is_row_counting_aggregate(agg),
                )
            )
    order = select.args.get("order")
    if order:
        for ordered in order.expressions:
            sig.order_columns.extend(
                ref
                for ref in _unique_column_refs_in_select(
                    ordered,
                    outer_aliases,
                    select,
                )
                if ref not in sig.order_columns
            )
            for agg in _aggregates_in_select(ordered, select):
                sig.measures.append(
                    MeasureSignature(
                        alias="ranking_metric",
                        function=agg.key.upper(),
                        source=_first_column_ref(agg, outer_aliases),
                        distinct=_is_distinct_agg(agg),
                        row_counting=_is_row_counting_aggregate(agg),
                    )
                )

    group = select.args.get("group")
    if group:
        for item in group.expressions:
            ref = _column_ref(item, outer_aliases)
            if ref:
                sig.groups.append(ref)
    where = select.args.get("where")
    if where:
        sig.filters.extend(
            _where_filters(where.this, outer_aliases, scope_select=select)
        )
        sig.scalar_aggregate_comparisons.extend(
            _scalar_aggregate_comparisons(where.this, outer_aliases, select)
        )
    for case in select.find_all(exp.Case):
        if _nearest_select(case) is not select:
            continue
        for branch in case.args.get("ifs") or []:
            condition = branch.args.get("this")
            if condition:
                sig.conditional_filters.extend(
                    _where_filters(condition, outer_aliases, scope_select=select)
                )
    for summed in select.find_all(exp.Sum):
        if _nearest_select(summed) is not select:
            continue
        argument = summed.this
        if argument and _looks_like_boolean_predicate(argument):
            sig.conditional_filters.extend(
                _where_filters(argument, outer_aliases, scope_select=select)
            )
    sig.has_order_limit = bool(
        select.args.get("order") is not None and select.args.get("limit") is not None
    )
    return sig


def _scalar_aggregate_comparisons(
    node: exp.Expression,
    outer_aliases: dict[str, str],
    outer_select: exp.Select,
) -> list[ScalarAggregateComparison]:
    """Extract predicates such as ``detail.cost < (SELECT AVG(cost) ...)``."""

    operators = {
        exp.LT: "<",
        exp.LTE: "<=",
        exp.GT: ">",
        exp.GTE: ">=",
        exp.EQ: "=",
        exp.NEQ: "!=",
    }
    comparisons: list[ScalarAggregateComparison] = []
    for comparison in node.walk():
        operator = next(
            (symbol for kind, symbol in operators.items() if isinstance(comparison, kind)),
            None,
        )
        if operator is None or not isinstance(comparison.left, exp.Column):
            continue
        subquery = comparison.right
        if not isinstance(subquery, exp.Subquery):
            continue
        inner_select = subquery.this
        if not isinstance(inner_select, exp.Select):
            continue
        aggregate = next(
            (
                item
                for item in inner_select.walk()
                if isinstance(item, (exp.Sum, exp.Avg, exp.Count, exp.Max, exp.Min))
                and _nearest_select(item) is inner_select
            ),
            None,
        )
        if aggregate is None:
            continue
        source = _column_ref(comparison.left, outer_aliases)
        if not source:
            continue
        aggregate_source = _first_column_ref(
            aggregate,
            _aliases_for_select(inner_select),
        )
        comparisons.append(
            ScalarAggregateComparison(
                source=source,
                operator=operator,
                aggregate_function=aggregate.key.upper(),
                aggregate_source=aggregate_source,
            )
        )
    return comparisons


def _root_select(tree: exp.Expression) -> exp.Select | None:
    if isinstance(tree, exp.Select):
        return tree
    return next(
        (
            select
            for select in tree.find_all(exp.Select)
            if _nearest_select(select.parent) is None
        ),
        None,
    )


def _nearest_select(node: exp.Expression | None) -> exp.Select | None:
    current = node
    while current is not None:
        if isinstance(current, exp.Select):
            return current
        current = current.parent
    return None


def _aliases_for_select(select: exp.Select) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in select.find_all(exp.Table):
        if _nearest_select(table) is not select:
            continue
        aliases[table.alias_or_name] = table.name
        aliases[table.name] = table.name
    return aliases


def _aggregates_in_select(
    node: exp.Expression,
    select: exp.Select,
) -> list[exp.Expression]:
    return [
        item
        for item in node.walk()
        if isinstance(item, (exp.Sum, exp.Avg, exp.Count, exp.Max, exp.Min))
        and _nearest_select(item) is select
    ]


def _unique_column_refs_in_select(
    node: exp.Expression,
    aliases: dict[str, str],
    select: exp.Select,
) -> list[str]:
    refs: list[str] = []
    for column in node.find_all(exp.Column):
        if _nearest_select(column) is not select:
            continue
        ref = _column_ref(column, aliases)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _find_aggregate(node: exp.Expression) -> exp.Expression | None:
    for expr in node.walk():
        if isinstance(expr, (exp.Sum, exp.Avg, exp.Count, exp.Max, exp.Min)):
            return expr
    return None


def _is_distinct_agg(node: exp.Expression) -> bool:
    return bool(node.args.get("distinct")) or isinstance(node.this, exp.Distinct)


def _is_row_counting_aggregate(node: exp.Expression) -> bool:
    if isinstance(node, exp.Count):
        # DISTINCT over an event-row identifier is still an event count rather
        # than a distinct-entity count. Target alignment is decided later once
        # the counted column and FK are known.
        return True
    if not isinstance(node, exp.Sum) or node.this is None:
        return False
    if _looks_like_boolean_predicate(node.this):
        return True
    case = node.this.find(exp.Case)
    if case is None:
        return False
    values = [
        branch.args.get("true")
        for branch in case.args.get("ifs") or []
    ]
    values.append(case.args.get("default"))
    numeric = [
        value
        for value in values
        if isinstance(value, exp.Literal) and not value.is_string
    ]
    return bool(numeric) and len(numeric) == len([value for value in values if value is not None])


def _looks_like_boolean_predicate(node: exp.Expression) -> bool:
    return any(
        isinstance(expr, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ, exp.Like, exp.Between, exp.In))
        for expr in node.walk()
    )


def _tables_in_expr(node: exp.Expression | None, aliases: dict[str, str]) -> set[str]:
    if not node:
        return set()
    tables = set()
    for col in node.find_all(exp.Column):
        table = col.table
        if table:
            tables.add(aliases.get(table, table))
    return tables


def _first_column_ref(node: exp.Expression, aliases: dict[str, str]) -> str | None:
    for col in node.find_all(exp.Column):
        return _column_ref(col, aliases)
    return None


def _column_ref(node: exp.Expression, aliases: dict[str, str]) -> str | None:
    if not isinstance(node, exp.Column):
        return None
    if not node.table:
        return node.name
    return f"{aliases.get(node.table, node.table)}.{node.name}"


def _where_filters(
    node: exp.Expression,
    aliases: dict[str, str],
    *,
    scope_select: exp.Select | None = None,
) -> list[FilterSignature]:
    filters: list[FilterSignature] = []
    for expr in node.walk():
        if scope_select is not None and _nearest_select(expr) is not scope_select:
            continue
        if isinstance(expr, (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ)):
            filters.extend(_comparison_filters(expr, aliases))
        elif isinstance(expr, exp.Like):
            parsed = _like_filter(expr, aliases)
            if parsed:
                filters.append(parsed)
        elif isinstance(expr, exp.Between):
            filters.extend(_between_filters(expr, aliases))
        elif isinstance(expr, exp.In):
            parsed = _in_filter(expr, aliases)
            if parsed:
                filters.append(parsed)
        elif (
            isinstance(expr, exp.Is)
            and isinstance(expr.expression, exp.Null)
            and not isinstance(expr.parent, exp.Not)
        ):
            ref = _column_ref(expr.this, aliases)
            if ref:
                filters.append(FilterSignature(ref, "IS NULL", None))
        elif (
            isinstance(expr, exp.Not)
            and isinstance(expr.this, exp.Is)
            and isinstance(expr.this.expression, exp.Null)
        ):
            ref = _column_ref(expr.this.this, aliases)
            if ref:
                filters.append(FilterSignature(ref, "IS NOT NULL", None))
    return filters


def _comparison_filters(expr: exp.Expression, aliases: dict[str, str]) -> list[FilterSignature]:
    left = expr.left
    right = expr.right
    operator = _comparison_operator(expr)
    if isinstance(right, exp.Literal):
        return _expr_literal_filters(left, operator.upper(), _literal_value(right), aliases)
    if isinstance(left, exp.Literal):
        return _expr_literal_filters(right, _reverse_operator(operator.upper()), _literal_value(left), aliases)
    return []


def _expr_literal_filters(
    node: exp.Expression,
    operator: str,
    value: object,
    aliases: dict[str, str],
) -> list[FilterSignature]:
    normalized = _normalized_column_ref(node, aliases)
    if normalized:
        ref, transform = normalized
        if transform == "YEAR":
            value = f"{value}-01-01"
            if operator in {"=", "EQ"}:
                operator = "YEAR_EQ"
        return [FilterSignature(ref, operator, value)]

    if not isinstance(value, (int, float)):
        return []
    difference_filters = _difference_literal_filters(node, operator, value, aliases)
    if difference_filters:
        return difference_filters
    div_filters = _division_literal_filters(node, operator, value, aliases)
    if div_filters:
        return div_filters
    refs = _unique_column_refs(node, aliases)
    if len(refs) < 2:
        return []
    if operator in {">", ">=", "GT", "GTE"}:
        return [FilterSignature(ref, "=", value + 1) for ref in refs]
    if operator in {"<", "<=", "LT", "LTE"}:
        filters = [FilterSignature(refs[0], "=", 0)]
        filters.extend(FilterSignature(ref, "=", max(abs(value) + 1, 1000)) for ref in refs[1:])
        return filters
    return []


def _difference_literal_filters(
    node: exp.Expression,
    operator: str,
    value: int | float,
    aliases: dict[str, str],
) -> list[FilterSignature]:
    """Materialize simple year differences with values that satisfy the predicate."""
    node = _unwrap_paren(node)
    if not isinstance(node, exp.Sub):
        return []

    left = _normalized_column_ref(node.this, aliases)
    right = _normalized_column_ref(node.expression, aliases)
    current_year = date.today().year
    strict_delta = 1 if operator in {">", "<", "GT", "LT"} else 0
    threshold = int(value)

    if left and right and left[1] == right[1] == "YEAR":
        right_year = 2000
        if operator in {">", ">=", "GT", "GTE"}:
            left_year = right_year + threshold + strict_delta
        elif operator in {"<", "<=", "LT", "LTE"}:
            left_year = right_year + threshold - strict_delta
        else:
            return []
        return [
            FilterSignature(left[0], "=", f"{left_year:04d}-01-01"),
            FilterSignature(right[0], "=", f"{right_year:04d}-01-01"),
        ]

    if right and right[1] == "YEAR" and _is_current_year_expression(node.this):
        if operator in {">", ">=", "GT", "GTE"}:
            birth_year = current_year - threshold - strict_delta
        elif operator in {"<", "<=", "LT", "LTE"}:
            birth_year = current_year - threshold + strict_delta
        else:
            return []
        return [FilterSignature(right[0], "=", f"{birth_year:04d}-01-01")]
    return []


def _is_current_year_expression(node: exp.Expression) -> bool:
    sql = node.sql(dialect="sqlite").lower()
    return "strftime" in sql and any(token in sql for token in ("'now'", "current_"))


def _division_literal_filters(
    node: exp.Expression,
    operator: str,
    value: int | float,
    aliases: dict[str, str],
) -> list[FilterSignature]:
    node = _unwrap_paren(node)
    if not isinstance(node, exp.Div):
        return []
    numerator_refs = _unique_column_refs(node.this, aliases)
    denominator_refs = _unique_column_refs(node.expression, aliases)
    if not numerator_refs or not denominator_refs:
        return []
    if operator in {">", ">=", "GT", "GTE"}:
        filters = [FilterSignature(ref, "=", value + 1) for ref in numerator_refs]
        filters.extend(FilterSignature(ref, "=", 1) for ref in denominator_refs)
        return filters
    if operator in {"<", "<=", "LT", "LTE"}:
        filters = [FilterSignature(ref, "=", 0) for ref in numerator_refs]
        filters.extend(FilterSignature(ref, "=", max(abs(value) + 1, 1000)) for ref in denominator_refs)
        return filters
    return []


def _like_filter(expr: exp.Like, aliases: dict[str, str]) -> FilterSignature | None:
    ref = _column_ref(expr.this, aliases)
    if not ref or not isinstance(expr.expression, exp.Literal):
        return None
    pattern = str(_literal_value(expr.expression))
    value = pattern.rstrip("%") or pattern.replace("%", "")
    return FilterSignature(ref, "=", value)


def _between_filters(expr: exp.Between, aliases: dict[str, str]) -> list[FilterSignature]:
    normalized = _normalized_column_ref(expr.this, aliases)
    if not normalized or not isinstance(expr.args.get("low"), exp.Literal):
        return []
    ref, transform = normalized
    value = _literal_value(expr.args["low"])
    if transform == "YEAR":
        value = f"{value}-01-01"
    return [FilterSignature(ref, ">=", value)]


def _in_filter(expr: exp.In, aliases: dict[str, str]) -> FilterSignature | None:
    normalized = _normalized_column_ref(expr.this, aliases)
    if not normalized:
        return None
    literals = [item for item in expr.expressions if isinstance(item, exp.Literal)]
    if not literals:
        return None
    ref, transform = normalized
    value = _literal_value(literals[0])
    if transform == "YEAR":
        value = f"{value}-01-01"
    return FilterSignature(ref, "=", value)


def _normalized_column_ref(node: exp.Expression, aliases: dict[str, str]) -> tuple[str, str | None] | None:
    ref = _column_ref(node, aliases)
    if ref:
        return ref, None
    if isinstance(node, (exp.Cast, exp.TsOrDsToTimestamp)):
        return _normalized_column_ref(node.this, aliases)
    if isinstance(node, exp.TimeToStr):
        fmt = node.args.get("format")
        inner = _normalized_column_ref(node.this, aliases)
        if inner and isinstance(fmt, exp.Literal) and _literal_value(fmt) == "%Y":
            return inner[0], "YEAR"
    if isinstance(node, exp.Substring):
        start = node.args.get("start")
        length = node.args.get("length")
        inner = _normalized_column_ref(node.this, aliases)
        if (
            inner
            and isinstance(start, exp.Literal)
            and isinstance(length, exp.Literal)
            and str(_literal_value(start)) == "1"
            and str(_literal_value(length)) == "4"
        ):
            return inner[0], "YEAR"
    if isinstance(node, exp.Paren):
        return _normalized_column_ref(node.this, aliases)
    return None


def _unwrap_paren(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _unique_column_refs(node: exp.Expression, aliases: dict[str, str]) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()
    for col in node.find_all(exp.Column):
        ref = _column_ref(col, aliases)
        if ref and ref not in seen:
            refs.append(ref)
            seen.add(ref)
    return refs


def _comparison_operator(expr: exp.Expression) -> str:
    if isinstance(expr, exp.EQ):
        return "="
    if isinstance(expr, exp.GT):
        return ">"
    if isinstance(expr, exp.GTE):
        return ">="
    if isinstance(expr, exp.LT):
        return "<"
    if isinstance(expr, exp.LTE):
        return "<="
    if isinstance(expr, exp.NEQ):
        return "!="
    return expr.key.upper()


def _literal_value(node: exp.Literal) -> object:
    if node.is_string:
        return node.this
    text = str(node.this)
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _reverse_operator(operator: str) -> str:
    return {
        ">": "<",
        ">=": "<=",
        "<": ">",
        "<=": ">=",
    }.get(operator, operator)
