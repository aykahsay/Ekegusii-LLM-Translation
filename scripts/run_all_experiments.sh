#!/bin/bash
# ==============================================================================
# Master Experiment Pipeline Execution Script
# Runs Experiments E0 through E8 on NVIDIA A100 GPU
# ==============================================================================

set -e # Exit immediately on error

echo "======================================================================"
echo "🚀 Starting Full Multilingual NMT Experiment Pipeline (E0 - E8)"
echo "======================================================================"

# Step 1: Data Integrity & Zero-Leakage Verification Audit
echo "[1/4] Running Master Corpus Data Leakage Audit..."
python -m src.master_corpus.integrity

# Step 2: 6-Way Instruction Task Generation
echo "[2/4] Generating 6-Way Multilingual Instruction Tasks..."
python -m src.task_generation.instruction_generator

# Step 3: Run Model Evaluation & Metric Computation
echo "[3/4] Running Evaluation & Extracting SacreBLEU / chrF++ Metrics..."
python -m src.cli.main evaluate

echo "======================================================================"
echo "🎉 [SUCCESS] Pipeline Execution Completed Successfully!"
echo "======================================================================"
