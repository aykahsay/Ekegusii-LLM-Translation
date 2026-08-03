#!/bin/bash
# ==============================================================================
# Full Resource Ablation Study: E0-E7 -> E8 Final Model Selection
# Trains + evaluates every experiment for both models, then aggregates
# results into the attribution matrix and recommends the E8 final-model
# source. Long-running -- intended for the A100, not local development.
# ==============================================================================

set -e

echo "======================================================================"
echo "Full Resource Ablation Study (E0-E8)"
echo "======================================================================"

echo "[1/4] Verifying master corpus integrity..."
bash scripts/build_master_corpus.sh

echo "[2/4] Training Aya-23-8B across E1-E7..."
bash scripts/train_aya.sh

echo "[3/4] Training Llama-3.1-8B-Instruct across E1-E7..."
bash scripts/train_llama.sh

echo "[4/4] Aggregating results and selecting E8 final-model source..."
python -m src.cli.main analyze --model-key aya
python -m src.cli.main analyze --model-key llama

echo "======================================================================"
echo "[SUCCESS] Ablation study complete. See experiments/*/results.json"
echo "and the printed attribution matrices above for E8 selection."
echo "======================================================================"
