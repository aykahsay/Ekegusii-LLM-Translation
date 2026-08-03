# API Reference

Entry points per module. Full parameter/return documentation lives in each
function's docstring (Google style, with `Args`/`Returns`/`Raises`) -- this
page is a map to find the right one, not a substitute for reading it.

## Data access

- `src.master_corpus.manager.MasterCorpusManager` -- load sentence/lexical
  corpora and the fixed train/val/test splits.
- `src.master_corpus.loader.UnifiedDatasetLoader` -- resolve a named dataset
  config (`configs/datasets/*.yaml`) straight to a DataFrame.
- `src.master_corpus.integrity.DataLeakageChecker` /
  `src.master_corpus.leakage.LeakageAuditReporter` -- verify and report on
  zero-leakage across splits.
- `src.master_corpus.statistics.CorpusStatistics` -- language/source/split
  descriptive statistics.
- `src.master_corpus.sampling.CorpusSampler` -- deterministic subsampling.

## Preprocessing

- `src.preprocessing.normalize.normalize_text` -- per-cell text normalization.
- `src.preprocessing.deduplicate.find_near_duplicates` -- fuzzy duplicate detection.
- `src.preprocessing.language_detection.matches_expected_language` -- English/Kiswahili
  LID; `ekegusii_lexicon_overlap` for Ekegusii (no trained LID model exists for it).
- `src.preprocessing.filtering.apply_filters` -- length + language-consistency filtering.

## Tokenizer analysis

- `src.tokenizer.aya.load_aya_tokenizer` / `src.tokenizer.llama.load_llama_tokenizer`
- `src.tokenizer.compare.TokenizerComparator` -- the notebook-04 entry point:
  `compare()`, `compare_vocabulary_coverage()`, `recommend_base_model()`.

## Task generation

- `src.task_generation.instruction_generator.InstructionTaskGenerator` -- 6-way
  sentence-level instruction tasks.
- `src.task_generation.lexical_tasks.LexicalTaskGenerator` -- dictionary-term tasks.
- `src.task_generation.prompt_templates.format_completion_prompt` /
  `format_chat_messages` -- single-sentence inference-time prompt formatting.

## Datasets & training

- `src.datasets.builder.InstructionDatasetBuilder` -- tokenize prompt/response
  pairs into a HF `Dataset` with label-masked prompts.
- `src.datasets.dataloader.build_train_dataloader` / `build_eval_dataloader`
- `src.models.common.run_qlora_training` -- the shared training loop.
- `src.models.aya.trainer.AyaTrainingPipeline` / `src.models.llama.trainer.LlamaTrainingPipeline`

## Evaluation

- `src.evaluation.sacrebleu.SacreBLEUEvaluator`, `.chrf.ChrFEvaluator`,
  `.comet.CometEvaluator` -- automatic metrics.
- `src.evaluation.human_eval.HumanEvalTemplateBuilder` / `HumanEvalAggregator`
- `src.evaluation.significance.PairedBootstrapTest`
- `src.evaluation.report.PublicationReportGenerator`

## Experiments

- `src.experiments.base.BaseExperiment` -- subclass for a new experiment;
  implement `build_training_tasks()`.
- `src.experiments.ablation.AblationAggregator` -- aggregate saved results,
  select the E8 final-model source.

## CLI

`python -m src.cli.main --help` for the full command list (`audit`,
`generate-tasks`, `evaluate`, `train`, `run-eval`, `translate`,
`schedule-preview`, `analyze`).
