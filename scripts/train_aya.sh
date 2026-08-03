#!/bin/bash
# ==============================================================================
# Aya-23-8B QLoRA Training: Experiments E1-E7
# Trains Aya on each resource-controlled experiment configuration in turn.
# Checkpoints land under checkpoints/aya/{EXPERIMENT_ID}/.
# Requires an NVIDIA A100 (or equivalent) GPU with bitsandbytes 4-bit support.
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
echo "Training Aya-23-8B: Experiments E1-E7"
echo "======================================================================"

for experiment_id in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------------------------------------"
    echo "Training aya on ${experiment_id}..."
    echo "----------------------------------------------------------------------"
    python -m src.cli.main train "${experiment_id}" --model-name aya
done

echo "======================================================================"
echo "[SUCCESS] Aya-23-8B training complete for all experiments."
echo "======================================================================"
