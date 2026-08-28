#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/runeval.sh <3b|1b>
# Run this script from the repo root.

SIZE="${1:-}"
case "$SIZE" in
  3b) PREFIX="Llama-3.2-3B" ;;
  1b) PREFIX="Llama-3.2-1B" ;;
  *)  echo "Tsage: $0 <3b|1b>" >&2; exit 1 ;;
esac

DATA_DIR="data"
MODEL_DIR="models_${SIZE}"
LOG_DIR="logits/${SIZE}"
RES_DIR="results_${SIZE}"

NGL=99
CTX=512
BATCH=2048
SEQ=4
THREADS=4

mkdir -p "$LOG_DIR" "$RES_DIR"
QUANTS=(Q6_K Q5_K_M Q4_K_M Q3_K_M Q2_K)

for LC in el en; do
  CORP="$DATA_DIR/flores_${LC}.txt"
  BASE="$LOG_DIR/${LC}_q8.kld"

  if [[ ! -f "$CORP" ]]; then
    echo "Error: Missing corpus $CORP (Run scripts/prep_data.py first)." >&2
    exit 1
  fi

  if [[ ! -f "$BASE" ]]; then
    echo ">> [${SIZE}/${LC}] building Q8_0 baseline <<"
    llama-perplexity -m "$MODEL_DIR/${PREFIX}.Q8_0.gguf" -f "$CORP" \
      -ngl "$NGL" -c "$CTX" -b "$BATCH" -np "$SEQ" -t "$THREADS" \
      --kl-divergence-base "$BASE" 2>&1 | tee "$RES_DIR/${LC}_Q8_0.txt"
  fi

  for Q in "${QUANTS[@]}"; do
    echo ">> [${SIZE}/${LC}] ${Q} vs Q8_0 <<"
    llama-perplexity -m "$MODEL_DIR/${PREFIX}.${Q}.gguf" -f "$CORP" \
      -ngl "$NGL" -c "$CTX" -b "$BATCH" -np "$SEQ" -t "$THREADS" \
      --kl-divergence-base "$BASE" --kl-divergence \
      2>&1 | tee "$RES_DIR/${LC}_${Q}.txt"
  done
done

echo "Model evaluation completed for ${SIZE}"
