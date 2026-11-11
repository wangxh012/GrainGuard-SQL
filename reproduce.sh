#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$ARTIFACT_DIR/code"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == "--verify-only" ]]; then
  "$PYTHON_BIN" "$ARTIFACT_DIR/scripts/verify_artifact.py"
  exit 0
fi

"$PYTHON_BIN" "$ARTIFACT_DIR/scripts/verify_artifact.py"

NATURAL_OUT="${NATURAL_OUT:-$CODE_DIR/artifacts/reproduced/natural_v23}"
CONTROLLED_OUT="${CONTROLLED_OUT:-$CODE_DIR/artifacts/reproduced/controlled_supported}"
BIRD_DB_DIR="$CODE_DIR/artifacts/datasets/bird/raw/dev_20240627/dev_databases"

if [[ ! -d "$BIRD_DB_DIR" ]]; then
  echo "Missing BIRD databases: $BIRD_DB_DIR" >&2
  echo "See README.md for dataset placement." >&2
  exit 2
fi

if [[ -e "$NATURAL_OUT" && -n "$(find "$NATURAL_OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty output directory: $NATURAL_OUT" >&2
  exit 2
fi

mkdir -p "$NATURAL_OUT/inputs" "$NATURAL_OUT/replay" "$NATURAL_OUT/effectiveness" \
  "$NATURAL_OUT/evidence_scope" "$NATURAL_OUT/review_audit"
cp "$ARTIFACT_DIR"/inputs/natural/*.jsonl \
  "$ARTIFACT_DIR/inputs/natural/v23_candidate_inputs.manifest.json" \
  "$NATURAL_OUT/inputs/"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" workflows/10_natural_candidate_v23/run_natural_candidate_replay_v23.py \
  --input-dir "$NATURAL_OUT/inputs" --output-dir "$NATURAL_OUT/replay" \
  --budgets 1,3,6,10,all
"$PYTHON_BIN" workflows/10_natural_candidate_v23/summarize_risk_tiers_v23.py \
  --traces "$NATURAL_OUT/replay/natural_candidate_v23.traces.jsonl" \
  --output-dir "$NATURAL_OUT/evidence_scope"
"$PYTHON_BIN" workflows/10_natural_candidate_v23/summarize_effectiveness_v23.py \
  --traces "$NATURAL_OUT/replay/natural_candidate_v23.traces.jsonl" \
  --input-manifest "$NATURAL_OUT/inputs/v23_candidate_inputs.manifest.json" \
  --output-dir "$NATURAL_OUT/effectiveness" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-20000}"
"$PYTHON_BIN" workflows/10_natural_candidate_v23/audit_abc_decisions_v23.py \
  --traces "$NATURAL_OUT/replay/natural_candidate_v23.traces.jsonl" \
  --output-dir "$NATURAL_OUT/review_audit"

"$PYTHON_BIN" workflows/09_executor_validation_v22/replay_controlled_supported.py \
  --input-dir "$ARTIFACT_DIR/inputs/controlled" --output-dir "$CONTROLLED_OUT"

echo "Reproduction complete."
echo "Natural results:   $NATURAL_OUT/effectiveness/v23_effectiveness.summary.json"
echo "Controlled results: $CONTROLLED_OUT/controlled_validation.json"
