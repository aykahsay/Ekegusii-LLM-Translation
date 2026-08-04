# Experiments

## The nine experiments

| ID | Name | Resources | Class |
|---|---|---|---|
| E0 | Baseline | None (zero-shot) | `src.experiments.baseline.BaselineExperiment` |
| E1 | English-Ekegusii | Bilingual eng<->eke only | `BilingualExperiment("eng_eke")` |
| E2 | Swahili-Ekegusii | Bilingual swa<->eke only | `BilingualExperiment("swa_eke")` |
| E3 | Bilingual | Both bilingual pairs combined | `BilingualExperiment("combined")` |
| E4 | Trilingual | Complete triplets, 6-way tasks | `src.experiments.trilingual.TrilingualExperiment` |
| E5 | Full Resources | Entire train split, unfiltered | `src.experiments.mono.FullResourcesExperiment` |
| E6 | Lexical Augmentation | E5 + lexical corpus tasks mixed | `src.experiments.lexical.LexicalAugmentationExperiment` |
| E7 | Curriculum Learning | E6 data, staged easy-to-hard | `src.experiments.curriculum.CurriculumLearningExperiment` |
| E8 | Final Model | Whichever of E0-E7 scores highest | Selected by `AblationAggregator.select_final_model`, then merged for deployment |

Each experiment trains BOTH Qwen2.5-7B-Instruct and Llama-3.1-8B independently, so the
ablation study answers two questions at once: which resource configuration
helps, and whether the answer differs by base model.

## Why `mono.py` implements E5, not a monolingual experiment

The filename is a scaffold artifact. `FullResourcesExperiment` additionally
exploits rows that `TrilingualExperiment` (E4) discards -- rows with only one
or two languages present, which never formed a complete triplet but still
contribute a valid pair. See the module docstring in
`src/experiments/mono.py` for the full reasoning.

## Running an experiment

```bash
python -m src.cli.main train E4_Trilingual --model-name qwen
python -m src.cli.main run-eval --model-name qwen --adapter-path checkpoints/qwen/E4_Trilingual/checkpoint-1500
```

Or the full sweep: `bash scripts/run_ablation.sh`.

## Augmentation is cross-cutting, not a numbered experiment

`src/experiments/augmentation.py`'s `augment_pairs_with_lexical_substitution`
is NOT one of E0-E8 -- it's an optional technique any experiment's pairs can
be passed through before tokenization, applying dictionary-term substitution
without needing a trained model (unlike back-translation, which needs one).
