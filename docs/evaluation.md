# Evaluation

## Automatic metrics

| Metric | Module | Notes |
|---|---|---|
| SacreBLEU | `src.evaluation.sacrebleu.SacreBLEUEvaluator` | Standardized, tokenizer-independent -- the metric of record |
| Raw NLTK BLEU | `src.evaluation.bleu.NLTKBleuEvaluator` | Secondary reference only, to demonstrate raw-BLEU incomparability |
| chrF++ | `src.evaluation.chrf.ChrFEvaluator` | Character n-gram F-score; more robust for morphologically rich Ekegusii |
| COMET | `src.evaluation.comet.CometEvaluator` | Source-aware neural metric; requires `unbabel-comet` + model download |
| Lexical accuracy | `src.evaluation.lexical_accuracy.compute_lexical_accuracy` | Exact + stem-prefix match against the lexical corpus |
| Rare-word accuracy | `src.evaluation.rare_word_accuracy.RareWordAccuracyEvaluator` | Splits test sentences into rare-word-containing vs. common, scores separately |
| Terminology consistency | `src.evaluation.terminology.TerminologyConsistencyChecker` | Curated institutional terms, self-consistency not exact-match |

## Human evaluation

`src.evaluation.human_eval.HumanEvalTemplateBuilder` samples 100+ test sentences
(stratified via `CorpusSampler`) and builds a blank-score template for native
speakers to fill in: **fluency**, **adequacy**, **cultural accuracy** (1-5 each).
`HumanEvalAggregator` summarizes completed scores per model/experiment and
tracks completion rate toward the 100-sentence requirement.

## Statistical significance

A difference in mean SacreBLEU/COMET between two experiments is not evidence of
improvement by itself. `src.evaluation.significance.PairedBootstrapTest` runs a
paired bootstrap resampling test (10,000 resamples by default) over
per-sentence scores from the SAME test set, and reports a one-sided p-value.
`src.experiments.ablation.AblationAggregator.run_significance_tests` runs this
across the meaningful adjacent E0-E7 comparisons automatically.

## Publication report

`src.evaluation.report.PublicationReportGenerator` combines the E0-E8
attribution matrix (`ResourceAttributionAnalyzer`), the human-eval summary, and
significance results into one Markdown report.

## What every score is measured against

All automatic metrics in this project score against `master_test.csv` -- the
single frozen 4,928-row held-out split -- regardless of which resources an
experiment trained on. See `docs/datasets.md` and `docs/reproducibility.md`.
