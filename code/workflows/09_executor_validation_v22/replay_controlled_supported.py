"""Replay frozen controlled SQL with the natural-candidate verification settings."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CODE), str(CODE / 'workflows/05_reporting')]

from grain_test_sql.contract import MetricContract
from grain_test_sql.evidence_scope import scoped_test_plans
from grain_test_sql.executor import execute_sql
from grain_test_sql.pair_builder import PairBuilder
from grain_test_sql.signature_extractor import extract_signature
from grain_test_sql.validator import validate_pair
from run_natural_budget_sensitivity import (
    SchemaRegistry, classify_at_budget, registry_with_filter_uniqueness_profile,
    trace_row,
)
from run_budget_sensitivity import ordered_pairs, VALUE_PROFILES

SPEC = importlib.util.spec_from_file_location(
    'controlled_independent_audit',
    CODE / 'workflows/07_controlled_validity/build_controlled_validity_audit.py',
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def dump_rows(path, rows):
    with path.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def builder_for(sql, registry, contract):
    signature = extract_signature(sql)
    builder = PairBuilder(
        registry, contract, filters=signature.filters,
        conditional_filters=signature.conditional_filters,
        required_tables=signature.tables,
        join_column_pairs=signature.join_column_pairs,
        group_columns=signature.groups,
        scalar_aggregate_comparisons=signature.scalar_aggregate_comparisons,
        projection_columns=signature.direct_projections,
        ranking_selection=signature.has_order_limit,
        ranking_count_sources=[m.source for m in signature.measures
                               if m.alias == 'ranking_metric' and m.function == 'COUNT'],
        enable_cgr_group_ranking_contrast=True,
    )
    return signature, builder


def reconstruct(row, registry, registry_path, trace, kind):
    sql = row['candidate sql']
    contract = MetricContract.from_dict(row['contract'])
    signature, builder = builder_for(sql, registry, contract)
    recorded = {plan.key: plan for plan in trace.plan_traces}
    for measure in contract.measures:
        for index, scoped in enumerate(scoped_test_plans(
            measure, signature, registry, output_group_ref=contract.output_group.ref,
        )):
            plan = scoped.plan
            identity = f'{measure.measure_id}:{plan.test_type}:{plan.child_table}:{index}'
            found = recorded.get(identity)
            if found is None or found.first_violation_rank is None:
                continue
            if kind == 'challenge' and scoped.evidence_scope != 'SUPPORTED':
                continue
            pairs = ordered_pairs(builder, plan, signature.filters, signature.conditional_filters)
            pair = pairs[found.first_violation_rank - 1]
            validation = validate_pair(sql, pair, registry).to_dict()
            if validation['status'] != 'VIOLATION':
                raise ValueError(f'witness replay disagrees with trace: {identity}')
            result = {
                'audit_id': '|'.join(map(str, [row['dataset'], row['db_id'], row['question_id'], row['contract_type'], row.get('contract_index', 0)])),
                'dataset': row['dataset'], 'operator': plan.test_type,
                'question_id': row['question_id'], 'db_id': row['db_id'],
                'contract_index': row.get('contract_index', 0),
                'registry_path': str(registry_path), 'contract': row['contract'],
                'candidate_sql': sql, 'measure_id': measure.measure_id,
                'plan': {'test_type': plan.test_type, 'child_table': plan.child_table},
                'pair_meta': pair.meta, 'd0': pair.d0, 'd1': pair.d1,
                'replayed_output_d0': execute_sql(sql, pair.d0, registry),
                'replayed_output_d1': execute_sql(sql, pair.d1, registry),
                'validation': validation, 'evidence_scope': scoped.evidence_scope,
            }
            result['independent_checks'] = AUDIT.independent_precheck(result)
            result['registry_path'] = registry_path.relative_to(CODE).as_posix()
            if kind == 'control':
                result['diagnostic_scope'] = 'deferred ranking output change; not a confirmed grain violation'
            return result
    raise ValueError(f'cannot reconstruct retained evidence: {row["dataset"]}/{row["question_id"]}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.input_dir / f'{kind}s.jsonl' for kind in ['challenge', 'control']]
    jobs = [(kind, row) for kind, path in zip(['challenge', 'control'], paths) for row in load(path)]
    if Counter(kind for kind, _ in jobs) != {'challenge': 580, 'control': 580}:
        raise ValueError('expected all 580 frozen challenges and all 580 controls')
    keys = [(kind, row['dataset'], row['db_id'], row['question_id'], row['contract_type'], row.get('contract_index', 0)) for kind, row in jobs]
    if len(set(keys)) != 1160:
        raise ValueError('duplicate controlled input')
    rows, witnesses, diagnostics, cache = [], [], [], {}
    for number, (kind, row) in enumerate(jobs, 1):
        dataset, db_id = row['dataset'], row['db_id']
        registry_path = CODE / f'artifacts/datasets/{dataset}/eval/registries/{db_id}.yaml'
        cache_key = (dataset, db_id)
        if cache_key not in cache:
            cache[cache_key] = SchemaRegistry.from_yaml(registry_path)
        registry = cache[cache_key]
        if dataset == 'bird':
            database = CODE / f'artifacts/datasets/bird/raw/dev_20240627/dev_databases/{db_id}/{db_id}.sqlite'
            if not database.is_file():
                raise FileNotFoundError(database)
            registry = registry_with_filter_uniqueness_profile(
                registry, database, row['candidate sql'], decision_eligible=False,
            )
        trace = trace_row(dataset, row, registry, output_comparator='exact')
        supported = classify_at_budget(trace, None, evidence_scope_policy='supported').status
        unrestricted = classify_at_budget(trace, None, evidence_scope_policy='unrestricted').status
        rows.append({
            'kind': kind, 'dataset': dataset, 'db_id': db_id,
            'question_id': row['question_id'], 'operator': row['contract_type'],
            'contract_index': row.get('contract_index', 0),
            'status': supported, 'unrestricted_status': unrestricted,
            'trace': asdict(trace),
        })
        if kind == 'challenge' and supported == 'VIOLATION':
            witnesses.append(reconstruct(row, registry, registry_path, trace, kind))
        if kind == 'control' and supported == 'ABSTAIN':
            diagnostics.append(reconstruct(row, registry, registry_path, trace, kind))
        if number % 100 == 0 or number == len(jobs):
            print(f'controlled replay {number}/{len(jobs)}', flush=True)
    counts = {kind: dict(Counter(r['status'] for r in rows if r['kind'] == kind)) for kind in ['challenge', 'control']}
    by_operator = {kind: {op: dict(Counter(r['status'] for r in rows if r['kind'] == kind and r['operator'] == op))
                         for op in sorted({r['operator'] for r in rows})} for kind in ['challenge', 'control']}
    valid = sum(w['independent_checks']['code_audit_valid'] for w in witnesses)
    summary = {
        'protocol': 'controlled_supported_aligned_20261111',
        'settings': {'evidence_scope_policy': 'supported', 'budget': 'all',
                     'value_profiles': list(VALUE_PROFILES), 'output_comparator': 'exact',
                     'bird_instance_uniqueness': 'pre-execution evidence scope only'},
        'inputs': {path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': 580} for path in paths},
        'counts': counts, 'by_operator': by_operator,
        'witness_audit': {'retained': len(witnesses), 'valid': valid,
                         'scope': 'independent schema, contract-value, perturbation-scope and replay code checks'},
    }
    (args.output_dir / 'controlled_validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    dump_rows(args.output_dir / 'controlled_supported_rows.jsonl', rows)
    dump_rows(args.output_dir / 'controlled_supported_witnesses.jsonl', witnesses)
    dump_rows(args.output_dir / 'controlled_deferred_controls.jsonl', diagnostics)
    table = ['kind\toperator\tn\tviolation\tpass\tabstain\tinvalid\tdecision_coverage_pct']
    for kind in ['challenge', 'control']:
        for op in ['all', *by_operator[kind]]:
            c = counts[kind] if op == 'all' else by_operator[kind][op]
            n = sum(c.values())
            table.append('\t'.join(map(str, [kind, op, n, c.get('VIOLATION', 0), c.get('PASS', 0), c.get('ABSTAIN', 0), c.get('INVALID_WITNESS', 0), f"{100 * (c.get('PASS', 0) + c.get('VIOLATION', 0)) / n:.2f}"])))
    (args.output_dir / 'controlled_validation.tsv').write_text('\n'.join(table) + '\n')
    print(json.dumps({'counts': counts, 'witness_audit': summary['witness_audit']}, indent=2))


if __name__ == '__main__':
    main()
