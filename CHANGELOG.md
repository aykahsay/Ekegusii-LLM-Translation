# Changelog

All notable changes to the `Ekegusii-LLM-Translation` research repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-03

### Added
- **Master Sentence Corpus (`data/master_corpus/master_sentence_corpus.csv`)**: 49,277 unique multilingual concepts spanning English, Kiswahili, and Ekegusii across 7 domain categories.
- **Master Lexical Corpus (`data/master_corpus/master_lexical_corpus.csv`)**: 268 dictionary entries for rare-word precision and terminology evaluation.
- **Leak-Proof Splits (`data/master_corpus/splits/`)**: Enforced 80% Train (39,421 concepts), 10% Validation (4,928 concepts), and 10% Test (4,928 concepts) splits with 0% data leakage guarantee.
- **Master Corpus Manager (`src/master_corpus/`)**: Python API (`manager.py`, `integrity.py`, `leakage.py`) providing zero-leakage validation and dataset loader interfaces.
- **6-Way Instruction Task Generator (`src/task_generation/`)**: Converts 49,277 concepts into 234,650 bidirectional translation instruction tasks (`ENG ↔ EKE`, `SWA ↔ EKE`, `ENG ↔ SWA`).
- **Tokenizer Fertility Analyzer (`src/tokenizer/`)**: Computes subword fragmentation rates, vocabulary coverage, and rare-word fertility for Qwen2.5-7B-Instruct and Meta Llama-3.1 8B.
- **QLoRA Fine-Tuning Engine (`src/models/`)**: Production training code using PEFT, Accelerate, and TRL with 4-bit quantization on NVIDIA A100 80GB GPU.
- **Multilingual MT Evaluation Suite (`src/evaluation/`)**: Automated evaluation of SacreBLEU, chrF++, Unbabel-COMET, exact dictionary accuracy, and morphological stem coverage.
- **Resource Attribution Matrix (`src/experiments/`)**: Systematic experiment execution suite evaluating Experiments E0 through E8 for hypothesis testing.
- **13 Numbered Reproducible Notebooks (`notebooks/`)**: Step-by-step notebooks (01 to 13) for interactive analysis, model training, and publication figure generation.
- **ACL/EMNLP Publication Assets (`paper/`, `docs/`)**: Documented architecture, LaTeX table exporters, and publication-quality figure generators.
