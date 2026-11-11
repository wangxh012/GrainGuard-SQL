# GrainGuard-SQL ICDE 2027 Artifact

## Access and permitted use

The original implementation and reproduction software may be made publicly
accessible and used, modified, and redistributed for noncommercial purposes
under the custom [GrainGuard-SQL Noncommercial License](LICENSE).
Commercial use is prohibited, including commercial product development,
paid services, internal commercial operations, and commercial model training
or evaluation. This is a noncommercial source-available release.
Permission is not limited to assigned reviewers.

BIRD and Spider benchmark material retains its upstream CC BY-SA 4.0 terms.
The software's noncommercial restriction does not apply to material governed
by those upstream licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for attribution, scope, and modification notices.

This directory is the submission artifact for the paper. It contains the frozen,
non-retired evidence used by the reported experiments, final human-label
exports, paper-facing result files, and the code needed to replay the main
natural-candidate and controlled-validation experiments.

## What is included

### Contracts, candidates, and final labels

| File | Data rows | Contents |
|---|---:|---|
| `data/contracts.jsonl` | 580 | Accepted question-grounded Contract V2.1 objects, including question, evidence, and full contract content |
| `data/candidates.jsonl` | 1,160 | Two frozen SQL candidates for each accepted contract |
| `data/human_labels.tsv` | 1,160 | Final natural-candidate outcomes with the A/B/C decision chain |
| `data/annotator_a_labels.jsonl` | 2,364 | A's final SQL-label decisions used by the experiments |
| `data/annotator_b_labels.jsonl` | 2,364 | B's final SQL-label decisions used by the experiments |
| `data/annotator_c_labels.jsonl` | 115 | C's actual SQL-label adjudications only |
| `data/contract_annotator_a_labels.jsonl` | 580 | A's final Contract V2.1 decisions |
| `data/contract_annotator_b_labels.jsonl` | 580 | B's final Contract V2.1 decisions |
| `data/contract_annotator_c_labels.jsonl` | 2 | C's actual Contract V2.1 adjudications only |
| `data/controlled_supported_witnesses.jsonl` | 337 | Retained alerts under the final verification settings, with independent checks |
| `inputs/controlled/challenges.jsonl` | 580 | Frozen controlled challenge SQL and contracts |
| `inputs/controlled/controls.jsonl` | 580 | Frozen, independently constructed synthetic negative controls and contracts |
| `data/controlled_witnesses.jsonl` | 365 | Archived unrestricted canonical-pair diagnostic witnesses |
| `data/matched_controls.jsonl` | 579 | Archived unrestricted canonical-pair completed controls |

The SQL A/B exports cover all 2,364 final, non-retired records cited by the
experiments: 2,190 primary records and 174 supplemental specification-failure
records. C is not padded with null records: its 115 rows are only cases that
required adjudication. The contract exports follow the same rule: A and B each
contain all 580 contracts, while C contains the two actual adjudications.

Every annotator export includes the benchmark question and categorical
decision fields. SQL records are keyed by `sample_id`, because a question may
have a reference SQL and multiple model SQL candidates.

For a contract label, locate the corresponding full contract in
`data/contracts.jsonl` using `(dataset, db_id, question_id)`. The label's
`contract_id` is the stable review identifier; the contracts file stores the
actual contract under its `contract` field.

Contract V2.1 is the final contract revision used by the V2.3 experiment;
these version numbers describe different components. The replay/audit scripts
read the same final `data/contract_annotator_{a,b,c}_labels.jsonl` exports,
filtered by dataset. No duplicate reviewer-label copies are included under
`code/`. The frozen review context (question, evidence, contract, and review
metadata) is retained under `data/contract_review/{bird,spider}/`; these are
audit inputs, not earlier annotation versions. In particular, the BIRD review
packets retain evidence and question wording not fully present in the compact
`data/contracts.jsonl` export.

The two manifests and their consolidated JSONL files under
`code/artifacts/annotations/sql_labels/` are the exact primary and supplemental
final SQL-label sources consumed by the audit scripts. They retain internal
decision-chain metadata needed for exact validation; use the six files under
`data/` when only reviewer decisions and questions are needed.

## Result files

| Files | Meaning |
|---|---|
| `results/natural_replay.{json,tsv}` | Raw aggregate output of the canonical natural-candidate replay |
| `results/natural_effectiveness.{json,tsv}` | Main precision, recall, F1, coverage, and uncertainty summaries |
| `results/evidence_scope.{json,tsv,jsonl}` | Evidence-scope summary, compact table, and per-candidate ledger |
| `results/human_label_audit.json` | A/B/C review coverage and consistency audit |
| `results/baseline_comparison.{json,tsv}` | Frozen shared-BIRD comparison, including the GLM-5.1 and Qwen3.6-27B contract-aware LLM judges |
| `results/baseline_coverage_audit.json` | Baseline input and coverage audit |
| `results/llm_judge_predictions/*.jsonl` | Eight frozen prediction files for prompt-only and contract-aware judges, crossed with two judge models and two BIRD candidate generators |
| `results/ablations.{json,tsv}` | Frozen ablation summary and paper-facing table |
| `results/controlled_validation.{json,tsv}` | Controlled results using the same supported policy and search as natural candidates |
| `results/controlled_supported_rows.jsonl` | All 1,160 controlled outcomes and search traces |
| `results/controlled_deferred_controls.jsonl` | Four deferred ranking-control pairs and replay outputs |
| `results/controlled_executor_diagnostic.{json,tsv}` | Archived unrestricted canonical-pair executor study |
| `results/canonical_source_manifest.json` | Source-run provenance for the packaged results |

JSON files are the detailed machine-readable summaries, TSV files are concise
paper-facing tables, and JSONL files are row-level ledgers. Paths recorded in
the canonical source manifest refer to the original repository layout; the
portable copies are the files listed above.
The final row-level label outcomes are provided once, in `data/human_labels.tsv`.

The contract-aware LLM controls receive the frozen metric contract and schema
metadata, but not GrainGuard-SQL decisions, executable witnesses, repairs, or
human labels. Their pooled 614-candidate rows are the two contract-aware judge
results reported in the manuscript; the same files also retain the two
307-candidate generator-pool breakdowns for auditability.

Both judge families reference their prediction files by paths relative to this
artifact directory in `results/baseline_comparison.json`. The prompt-only judges
receive the reference-free linked schema but no metric contract. Full frozen
prediction files are retained; scoring selects the same 307 questions per
generator using the explicit omitted-question lists in the summary.

The main natural result recorded in `manifest.json` is 86 TP, 17 FP, and 25 FN:
precision 83.50%, recall 77.48%, F1 80.37%, and decision coverage 94.83%.

**Evaluation scope.** The paper reports candidate-level effectiveness on 1,160
SQL candidates associated with 580 accepted contracts out of 817 questions.
Decision coverage of 94.83% is conditional on contract acceptance. The result
files additionally retain a gate-inclusive analysis over 1,634 candidates,
treating contract-construction failures as non-alerts; this yields 44.79% recall,
58.31% F1, and 67.32% decision coverage.

## Integrity check

From this directory, run:

```bash
./reproduce.sh --verify-only
```

`manifest.json` records expected row counts for the principal datasets and
SHA-256 checksums for the packaged code, scripts, dependencies declaration,
documentation, data, results, and final-label provenance files. This includes
the dynamically loaded controlled-validity audit helper. The verification
command does not require benchmark databases.
It also recomputes all twelve LLM judge summary rows (four methods, each with
two generator pools and one pooled row) from the packaged predictions,
`inputs/natural/`, and `data/human_labels.tsv`, and checks JSON/TSV consistency.
To run only this offline score verification, use:

```bash
python3 scripts/verify_llm_judges.py
```

## Full reproduction

The artifact was checked with Python 3.9.6, PyYAML 6.0.3, and sqlglot 30.11.0.
Install the declared compatible dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The official BIRD development databases cannot be redistributed here. Place
them at:

```text
code/artifacts/datasets/bird/raw/dev_20240627/dev_databases/<db_id>/<db_id>.sqlite
```

Then run:

```bash
./reproduce.sh
```

This verifies the package, replays all 1,160 natural candidates, recomputes the
natural effectiveness/evidence-scope/reviewer-audit summaries, and runs the 580
controlled challenge/control pairs. Outputs are written to
`code/artifacts/reproduced/`. No LLM API call is made. The script refuses to
overwrite a non-empty natural output directory; set `NATURAL_OUT` and
`CONTROLLED_OUT` to choose other locations.

Controlled and natural-candidate experiments use the `supported` policy,
the same catalogue of up to ten deduplicated numeric profiles, BIRD
candidate-specific uniqueness profiling, and exact output comparison.
Controlled replay retains all 580 challenges and all 580 controls. Challenges
yield 337 confirmed alerts, 182 PASS, 37 ABSTAIN, and 24 INVALID_WITNESS;
controls yield 575 PASS, four ABSTAIN, and one INVALID_WITNESS, with no
confirmed false alerts. Independent schema, contract-value, perturbation-scope,
and replay checks accept 336 of the 337 retained witnesses. The remaining
multihop CGR pair changes `disp`, an intermediate relation outside the audit's
admitted table set. All outcomes remain in the released files.

The four deferred controls have ranking-only evidence. Numeric offsets change
the number-dependent sum after event splitting. In three cases the winner is
unchanged but its projected metric changes; in the other the winner changes.
These outcomes are recorded as ABSTAIN, with their pairs and outputs under
`results/controlled_deferred_controls.jsonl`.

The earlier unrestricted canonical-pair executor study is retained under
`results/controlled_executor_diagnostic.*` and the two archived data files above.
It has 365 triggers (including ten TST cases), 191 PASS, and 24 invalid
challenges; 579 controls complete without a trigger. These diagnostic counts
are separate from the final supported controlled results.

The LLM baseline scores are recomputed from frozen predictions without new API
calls. Other baseline and ablation files are frozen reported summaries; their
historical exploratory runs and caches are not required by the main replay.

## Paper-to-artifact map

- Contract construction and content: `data/contracts.jsonl`
- Natural candidates and their final outcomes: `data/candidates.jsonl` and
  `data/human_labels.tsv`
- Complete final SQL decisions by reviewer:
  `data/annotator_{a,b,c}_labels.jsonl`
- Complete final Contract V2.1 decisions by reviewer:
  `data/contract_annotator_{a,b,c}_labels.jsonl`
- Main natural metrics and scope analysis: `results/natural_effectiveness.*`,
  `results/natural_replay.*`, and `results/evidence_scope.*`
- Baselines and ablations: `results/baseline_*` and `results/ablations.*`
- Controlled validation: `inputs/controlled/{challenges,controls}.jsonl`,
  `data/controlled_supported_witnesses.jsonl`,
  `results/controlled_supported_rows.jsonl`, and `results/controlled_validation.*`
- Implementation: `code/grain_test_sql/`
- Reproduction entry point: `reproduce.sh`

## Additional requirements

- The official BIRD database files are not included. They must be obtained
  separately from the BIRD benchmark for full reproduction.
- Original implementation and reproduction software is subject to `LICENSE`.
  Benchmark material is separately described in `THIRD_PARTY_NOTICES.md`.
