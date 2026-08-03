# Reproducibility

## The one rule that makes everything else valid

`data/master_corpus/splits/master_{train,val,test}.csv` are frozen. They must
never be regenerated. Every experiment's `build_test_pairs()`
(`src/experiments/base.py`) reads `master_test.csv` unconditionally --
experiment-specific resource configuration only ever changes
`build_training_tasks()`. `CorpusSplitter.assert_splits_not_overwritten()`
raises `SplitImmutabilityError` if anything tries to regenerate them.

If you ever need a NEW derived subset (e.g. a per-direction validation file),
build it via `CorpusSplitter.project_to_existing_split()`, which partitions
using the EXISTING frozen concept_id -> split assignment rather than
re-splitting independently -- the latter risks a concept appearing in both a
derived train file and the master test file.

## Seeding

`src.utils.seed.set_seed(42)` seeds Python `random`, NumPy, and PyTorch
(CPU + all CUDA devices), sets `PYTHONHASHSEED`, and requests deterministic
CUDA algorithms. Called at the top of every training entry point
(`AyaTrainingPipeline.run`, `LlamaTrainingPipeline.run`, `src.cli.train.run_train`).

## Environment

`requirements.txt` pins minimum versions for the full stack (torch, transformers,
peft, trl, accelerate, bitsandbytes, hydra-core/omegaconf, sacrebleu,
unbabel-comet, ...). `environment.yml` provides the conda equivalent.
Training requires an NVIDIA GPU with `bitsandbytes` 4-bit support (developed
against an A100 80GB); everything through task generation and tokenizer
analysis runs on CPU.

## From-scratch reproduction

```bash
bash scripts/reproduce_paper.sh
```

Runs, in order: corpus integrity audit -> instruction task generation ->
tokenizer analysis -> full E0-E8 ablation study (both models) -> publication
figure notebooks. Individual stages are also runnable standalone (see
`scripts/*.sh`) for iterating on one part without re-running the whole pipeline.

## Config provenance

`src.utils.config.save_resolved_config()` writes the fully-resolved
(merged, no remaining references) configuration for a training run to disk
alongside its checkpoints, so a checkpoint can always be traced back to the
exact hyperparameters that produced it, independent of later edits to the
source YAML files.
