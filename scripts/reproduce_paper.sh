#!/bin/bash
# ==============================================================================
# Full Paper Reproduction Pipeline
# Runs every stage needed to go from raw master corpus to publication
# figures/tables: integrity audit -> task generation -> tokenizer analysis
# -> full E0-E8 ablation study -> notebook-driven figure generation.
# This is the top-level script referenced in the paper's reproducibility
# statement. Requires an NVIDIA A100 (or equivalent) GPU for the training
# stage; everything before it can run on CPU.
# ==============================================================================

set -e

echo "======================================================================"
echo "Ekegusii-LLM-Translation: Full Paper Reproduction Pipeline"
echo "======================================================================"

echo "[1/5] Master corpus integrity verification..."
bash scripts/build_master_corpus.sh

echo "[2/5] Generating 6-way multilingual instruction tasks..."
bash scripts/generate_tasks.sh

echo "[3/5] Tokenizer fertility & vocabulary coverage analysis..."
bash scripts/tokenizer_analysis.sh

echo "[4/5] Running full E0-E8 resource ablation study (Qwen + Mistral)..."
bash scripts/run_ablation.sh

echo "[5/5] Executing publication-figure notebooks..."
for notebook in notebooks/09_translation_evaluation.ipynb notebooks/11_ablation_study.ipynb notebooks/12_error_analysis.ipynb notebooks/13_publication_figures.ipynb; do
    echo "  Executing ${notebook}..."
    jupyter nbconvert --to notebook --execute --inplace "${notebook}"
done

echo "======================================================================"
echo "[SUCCESS] Paper fully reproduced."
echo "Figures: paper/figures/ | Tables: paper/tables/ | Results: experiments/*/results.json"
echo "======================================================================"
