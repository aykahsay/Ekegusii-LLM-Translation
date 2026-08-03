#!/bin/bash
# ==============================================================================
# 6-Way Multilingual Instruction Task Generation
# Generates translation instruction tasks for the train/val/test splits.
# ==============================================================================

set -e

echo "======================================================================"
echo "Generating 6-Way Multilingual Instruction Tasks"
echo "======================================================================"

python -m src.cli.main generate-tasks

echo "======================================================================"
echo "[SUCCESS] Instruction tasks generated."
echo "======================================================================"
