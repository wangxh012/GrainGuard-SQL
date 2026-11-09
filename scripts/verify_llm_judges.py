#!/usr/bin/env python3
"""Recompute all reported LLM judge rows from packaged predictions and labels."""
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row):
    return str(row['db_id']), int(row['question_id'])


def compact(sql):
    return re.sub(r'\s+', '', sql)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    summary = json.loads((ROOT / 'results/baseline_comparison.json').read_text())
    with (ROOT / 'data/human_labels.tsv').open(newline='') as handle:
        labels = {(r['model'], *key(r)): r['final_positive'].lower() == 'true'
                  for r in csv.DictReader(handle, delimiter='\t') if r['dataset'] == 'bird'}
    reported = {(r['pool'], r['method']): r for r in summary['rows']}
    with (ROOT / 'results/baseline_comparison.tsv').open(newline='') as handle:
        table = {(r['pool'], r['method']): r
                 for r in csv.DictReader(handle, delimiter='\t')}
    cohorts = {r['pool']: r for r in summary['cohorts']}
    methods = dict(summary['llm_self_check_artifacts'])
    methods.update(summary['contract_aware_llm_judge_artifacts'])
    for method, paths in methods.items():
        totals = [0, 0, 0, 0, 0, 0]
        for pool, relative in paths.items():
            path = (ROOT / relative).resolve()
            require(ROOT in path.parents, 'Prediction path must be inside package')
            predictions = read_rows(path)
            indexed = {key(r): r for r in predictions}
            require(len(indexed) == len(predictions), 'Duplicate predictions: ' + relative)
            candidates = read_rows(ROOT / ('inputs/natural/' + pool + '_candidate.v23.jsonl'))
            omitted = set(cohorts[pool]['omitted_question_ids'])
            selected = [r for r in candidates if int(r['question_id']) not in omitted]
            require(len(selected) == cohorts[pool]['intersection_questions'], 'Wrong cohort size')
            counts = [0, 0, 0, 0, 0, 0]  # TP, FP, FN, TN, decided, N
            candidate_model = 'glm-5.1' if pool == 'bird_glm' else 'qwen3.6-27b'
            for candidate in selected:
                row = indexed[key(candidate)]
                require(compact(row['candidate_sql']) == compact(candidate['candidate sql']),
                        'Candidate SQL mismatch: ' + str(key(candidate)))
                positive = labels[(candidate_model, *key(candidate))]
                if method.startswith('llm_self_check_'):
                    policy = row['input_policy']
                    require(policy.get('schema_source') == 'contract_schema_linking'
                            and policy.get('reference_sql_derived_table_scope') is False
                            and policy.get('metric_grain_contract') is False,
                            'Unexpected prompt-only input policy')
                    decision = row['llm_self_check'].get('has_grain_risk')
                    predicted = decision is True
                    decided = isinstance(decision, bool) and not row.get('parse_error')
                else:
                    require(row['contract'] == candidate['contract'], 'Contract mismatch')
                    decision = row['contract_aware_judge'].get('decision')
                    predicted = decision == 'VIOLATION'
                    decided = decision in ('PASS', 'VIOLATION')
                index = (0 if predicted else 2) if positive else (1 if predicted else 3)
                counts[index] += 1
                counts[4] += int(decided)
                counts[5] += 1
            check(reported[(pool, method)], counts)
            totals = [a + b for a, b in zip(totals, counts)]
        row = reported[('bird_pooled', method)]
        check(row, totals)
        for pool in (*paths, 'bird_pooled'):
            result = reported[(pool, method)]
            entry = table[(pool, method)]
            for field in ('n', 'positive', 'tp', 'fp', 'fn', 'tn'):
                require(int(entry[field]) == result[field], 'TSV count mismatch')
            for field in ('precision', 'recall', 'f1', 'coverage'):
                require(entry[field + '_pct'] == '{:.2f}'.format(100 * result[field]),
                        'TSV metric mismatch')
        print('{}: TP/FP/FN={}/{}/{}, F1={:.2f}, verified'.format(
            row['method_label'], *totals[:3], row['f1'] * 100))
    print('All 12 LLM judge summary rows verified without API calls or databases.')


def check(row, counts):
    tp, fp, fn, tn, decided, n = counts
    expected = dict(tp=tp, fp=fp, fn=fn, tn=tn, n=n, positive=tp+fn,
                    precision=tp/(tp+fp), recall=tp/(tp+fn),
                    f1=2*tp/(2*tp+fp+fn), coverage=decided/n)
    for field, value in expected.items():
        require(math.isclose(row[field], value, rel_tol=1e-12, abs_tol=1e-12),
                '{} / {}: {} mismatch'.format(row['pool'], row['method'], field))


if __name__ == '__main__':
    main()
