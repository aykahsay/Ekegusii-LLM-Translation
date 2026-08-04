#!/bin/bash
# ==============================================================================
# Automatic Metric Evaluation: All Trained Checkpoints
# Runs SacreBLEU/chrF/lexical-accuracy evaluation for both models against
# the fixed master test split, for the core English<->Ekegusii direction.
# Adjust ADAPTER_PATHS below to point at your actual trained checkpoints
# once training (train_qwen.sh / train_llama.sh) has produced them.
# ==============================================================================

set -e

echo "======================================================================"
echo "Evaluating Qwen2.5-7B-Instruct and Llama-3.1-8B (zero-shot baseline, E0)"
echo "======================================================================"

python -m src.cli.main run-eval --model-name qwen --source-lang English --target-lang Ekegusii
python -m src.cli.main run-eval --model-name llama --source-lang English --target-lang Ekegusii

echo ""
echo "To evaluate a TRAINED checkpoint instead of the zero-shot baseline,"
echo "pass --adapter-path, e.g.:"
echo "  python -m src.cli.main run-eval --model-name qwen --adapter-path checkpoints/qwen/E4_Trilingual/checkpoint-1500"

echo "======================================================================"
echo "[SUCCESS] Evaluation complete."
echo "======================================================================"
