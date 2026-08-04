# Architecture

## Layer overview

```text
configs/            Layered YAML config groups (models, training, datasets, generation, prompts)
data/master_corpus/  The frozen master sentence/lexical corpora + splits
src/
  master_corpus/     Loading, validation, cleaning, splitting, statistics, leakage auditing, scheduling
  preprocessing/      Per-cell text normalization, dedup, language ID, filtering, export
  tokenizer/          Qwen/Llama tokenizer loaders + fertility/vocabulary/rare-word analysis
  task_generation/    Instruction-task construction (sentence-level, lexical, multilingual-mixed)
  datasets/           Tokenization -> HF Dataset, collation, weighted sampling, DataLoader factory
  models/             QLoRA model loading/training/inference/saving (Qwen + Llama)
  evaluation/         SacreBLEU, chrF, COMET, lexical/rare-word/terminology accuracy, significance, reports
  experiments/        E0-E8 experiment definitions + ablation aggregation
  visualization/      Publication-quality matplotlib figures
  cli/                Typer CLI wiring all of the above together
  utils/              Constants, config loading, logging, seeding, checkpointing, generic helpers
experiments/         Per-experiment output directories (results.json, checkpoints pointer)
checkpoints/         PEFT adapter checkpoints per model/experiment
outputs/             Logs, metrics, figures, tables, predictions
paper/               Manuscript, figures, tables, appendix
```

## Design principle: config composition, not duplication

`src/utils/config.py` composes YAML layers via `OmegaConf.merge`:

```text
configs/models/common.yaml -> configs/models/{model}_8b.yaml
configs/training/qlora.yaml -> configs/training/{model}_8b_qlora.yaml
```

Later layers override earlier ones key-for-key. A per-model file only needs to
specify what differs from the shared default -- this is why `configs/models/qwen_7b.yaml`
doesn't repeat `hardware`/`seed`/`caching` settings that live in `common.yaml`.

## Design principle: one shared implementation per cross-cutting concern

Qwen2.5-7B-Instruct and Llama-3.1-8B are trained, checkpointed, and evaluated through an
**identical** QLoRA pipeline -- only the base model ID and hyperparameters differ.
`src/models/common.py` holds that shared logic once; `src/models/qwen/*.py` and
`src/models/llama/*.py` are thin wrappers passing their own `MODEL_ID`. The same
pattern applies to `src/experiments/base.py` (shared experiment scaffold) and
`src/visualization/palette.py` (shared color system).

## Design principle: the master test split is immutable

Every experiment (E0-E8) evaluates against the exact same
`data/master_corpus/splits/master_test.csv`. `src/experiments/base.py`'s
`build_test_pairs()` always reads from this fixed split; experiment-specific
resource configuration only affects `build_training_tasks()`. This is what
makes cross-experiment comparison in the ablation study valid -- see
`docs/reproducibility.md`.

## The experiment inheritance chain

```text
BaselineExperiment (E0)         -- zero-shot, no training
BilingualExperiment (E1/E2/E3)  -- parametrized by mode: eng_eke | swa_eke | combined
TrilingualExperiment (E4)       -- complete triplets only, 6-way tasks
FullResourcesExperiment (E5)    -- entire train split, unfiltered, 6-way tasks
  -> LexicalAugmentationExperiment (E6)   -- E5's tasks + lexical corpus tasks mixed in
    -> CurriculumLearningExperiment (E7)  -- E6's tasks, reordered easy-to-hard
```

E8 (Final Model) is not a distinct training configuration -- `AblationAggregator.select_final_model()`
picks whichever of E0-E7 scores highest and that checkpoint gets merged
(`src.models.common.merge_and_save_adapter`) for deployment.
