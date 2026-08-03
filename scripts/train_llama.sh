#!/bin/bash
# ==============================================================================
# Llama-3.1-8B-Instruct QLoRA Training: Experiments E1-E7
# Trains Llama on each resource-controlled experiment configuration in turn.
# Checkpoints land under checkpoints/llama/{EXPERIMENT_ID}/.
# Requires an NVIDIA A100 (or equivalent) GPU and Hub access to
# meta-llama/Meta-Llama-3.1-8B-Instruct (accept Meta's license on the Hub first).
# ==============================================================================

set -e

EXPERIMENTS=(
    "E1_English_Ekegusii"
    "E2_Swahili_Ekegusii"
    "E3_Bilingual"
    "E4_Trilingual"
    "E5_Full_Resources"
    "E6_Lexical_Augmentation"
    "E7_Curriculum_Learning"
)

echo "======================================================================"
echo "Training Llama-3.1-8B-Instruct: Experiments E1-E7"
echo "======================================================================"

for experiment_id in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Training llama on ${experiment_id}..."
    echo "----------------------------------------------------------------------"
    python -m src.cli.main train "${experiment_id}" --model-name llama
done

echo "======================================================================"
echo "[SUCCESS] Llama-3.1-8B-Instruct training complete for all experiments."
echo "======================================================================"
