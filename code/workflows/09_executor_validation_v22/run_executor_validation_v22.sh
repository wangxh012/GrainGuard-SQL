#!/usr/bin/env bash
set -euo pipefail

# Executor/validation V2.2 only.  This script creates a new run directory and
# never writes the frozen Contract V2.1 inputs or paper-facing V2.1 artifacts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/runs/v22_executor_validation}"
BIRD_EVAL="${BIRD_EVAL:-artifacts/datasets/bird/eval/bird_eval_set.jsonl}"
SPIDER_EVAL="${SPIDER_EVAL:-artifacts/datasets/spider/eval/spider_eval_set.jsonl}"
BIRD_V21="${BIRD_V21:-artifacts/runs/bird/shared/00_generation_inputs/contract_v21_recovery/auto_contracts.v21.candidate.jsonl}"
SPIDER_V21="${SPIDER_V21:-artifacts/runs/spider/shared/00_generation_inputs/contract_v21_recovery/auto_contracts.v21.candidate.jsonl}"

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" workflows/09_executor_validation_v22/snapshot_v21_contracts.py \
  --bird-contracts "$BIRD_V21" --spider-contracts "$SPIDER_V21" --output-dir "$OUTPUT_DIR"

generate_dataset() {
  local dataset="$1" eval_set="$2" contracts="$3"
  local data_dir="$OUTPUT_DIR/$dataset"
  mkdir -p "$data_dir"
  # Deliberately absent: --require-gold-pass, --detectable-only, and
  # --prefer-validated-violation.  Their use would invalidate V2.2.
  "$PYTHON_BIN" workflows/02_validation/generate_bird_false_candidates.py \
    --eval-set "$eval_set" --auto-contracts "$contracts" --auto-contract-row-set \
    --contract-source auto --mutation-mode gold-sql-with-fallback --force-synthetic-fallback \
    --out "$data_dir/faults.legacy-generated.jsonl"
  "$PYTHON_BIN" workflows/09_executor_validation_v22/complete_multihop_tst_faults.py \
    --eval-set "$eval_set" --contracts "$contracts" \
    --faults "$data_dir/faults.legacy-generated.jsonl" \
    --out "$data_dir/faults.unfiltered.jsonl"
}

validate_dataset() {
  local dataset="$1" eval_set="$2" contracts="$3"
  local data_dir="$OUTPUT_DIR/$dataset"
  "$PYTHON_BIN" workflows/02_validation/run_bird_grain_repair.py \
    --eval-set "$eval_set" --auto-contracts "$contracts" --auto-contract-row-set \
    --contract-source auto --only-with-contract --only-predicted-contracts \
    --predictions "$data_dir/faults.unfiltered.jsonl" --skip-real-result-compare \
    --evidence-scope-policy unrestricted \
    --out "$data_dir/fault_validation.unfiltered.jsonl"
}

generate_dataset bird "$BIRD_EVAL" "$OUTPUT_DIR/inputs/bird.auto_contracts.v21.jsonl"
generate_dataset spider "$SPIDER_EVAL" "$OUTPUT_DIR/inputs/spider.auto_contracts.v21.jsonl"
"$PYTHON_BIN" workflows/09_executor_validation_v22/freeze_unfiltered_fault_pool.py \
  --contract-manifest "$OUTPUT_DIR/v21_contract_freeze.manifest.json" \
  --bird-faults "$OUTPUT_DIR/bird/faults.unfiltered.jsonl" \
  --spider-faults "$OUTPUT_DIR/spider/faults.unfiltered.jsonl" \
  --out "$OUTPUT_DIR/unfiltered_fault_pool.manifest.json"
validate_dataset bird "$BIRD_EVAL" "$OUTPUT_DIR/inputs/bird.auto_contracts.v21.jsonl"
validate_dataset spider "$SPIDER_EVAL" "$OUTPUT_DIR/inputs/spider.auto_contracts.v21.jsonl"

DATASET_CONFIG="{\"bird\":{\"eval\":\"$BIRD_EVAL\",\"faults\":\"$OUTPUT_DIR/bird/faults.unfiltered.jsonl\",\"results\":\"$OUTPUT_DIR/bird/fault_validation.unfiltered.jsonl\"},\"spider\":{\"eval\":\"$SPIDER_EVAL\",\"faults\":\"$OUTPUT_DIR/spider/faults.unfiltered.jsonl\",\"results\":\"$OUTPUT_DIR/spider/fault_validation.unfiltered.jsonl\"}}"
"$PYTHON_BIN" workflows/09_executor_validation_v22/build_unfiltered_control_audit.py \
  --dataset-config-json "$DATASET_CONFIG" --output-dir "$OUTPUT_DIR"
"$PYTHON_BIN" workflows/09_executor_validation_v22/certify_controls_v22.py \
  --controls "$OUTPUT_DIR/controls.unfiltered.jsonl" \
  --out "$OUTPUT_DIR/control_certification.tsv"
CERTIFIED_CONTROL_CONFIG="{\"bird\":{\"eval\":\"$BIRD_EVAL\",\"contracts\":\"$OUTPUT_DIR/inputs/bird.auto_contracts.v21.jsonl\"},\"spider\":{\"eval\":\"$SPIDER_EVAL\",\"contracts\":\"$OUTPUT_DIR/inputs/spider.auto_contracts.v21.jsonl\"}}"
"$PYTHON_BIN" workflows/09_executor_validation_v22/build_certified_control_pool_v22.py \
  --dataset-config-json "$CERTIFIED_CONTROL_CONFIG" \
  --out "$OUTPUT_DIR/certified_control_pool.jsonl"
"$PYTHON_BIN" workflows/09_executor_validation_v22/summarize_executor_validation_v22.py \
  --manifest "$OUTPUT_DIR/v21_contract_freeze.manifest.json" \
  --fault-pool-manifest "$OUTPUT_DIR/unfiltered_fault_pool.manifest.json" \
  --bird-fault-results "$OUTPUT_DIR/bird/fault_validation.unfiltered.jsonl" \
  --spider-fault-results "$OUTPUT_DIR/spider/fault_validation.unfiltered.jsonl" \
  --controls "$OUTPUT_DIR/controls.unfiltered.jsonl" \
  --control-certification "$OUTPUT_DIR/control_certification.tsv" \
  --certified-control-pool "$OUTPUT_DIR/certified_control_pool.jsonl" \
  --witness-audit "$OUTPUT_DIR/witness_audit.unfiltered.jsonl" \
  --output-dir "$OUTPUT_DIR"
"$PYTHON_BIN" workflows/09_executor_validation_v22/triage_invalid_witnesses_v22.py \
  --dataset-config-json "$DATASET_CONFIG" \
  --out "$OUTPUT_DIR/invalid_witness_triage.json"

echo "V2.2 summary: $OUTPUT_DIR/v22_executor_validation.summary.json"
echo "V2.2 table:   $OUTPUT_DIR/v22_executor_validation.table.tsv"
