# 🌍 Ekegusii-LLM-Translation: Resource-Aware Adaptation of Multilingual Large Language Models for Low-Resource Machine Translation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-PEFT-orange.svg)](https://huggingface.co/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Official open-source research repository for the ACL/EMNLP paper:
> **"Resource-Aware Adaptation of Multilingual Large Language Models for Low-Resource Machine Translation: A Case Study on Ekegusii"**

---

## 📌 Research Objective & Central Question

Low-resource African language translation faces severe data scarcity and morphological complexity. This repository provides a **Resource-Aware Instruction-Tuning Translation Framework** that systematically evaluates how different multilingual data resources (monolingual, bilingual, trilingual, and dictionary lexicons) contribute to machine translation performance when instruction-tuning Large Language Models (**Cohere Aya 23 8B** and **Meta Llama 3.1 8B**) on **Ekegusii (Bantu, eke)**, **Kiswahili (swh)**, and **English (eng)**.

> **Central Research Question**: *How can multilingual LLMs be effectively adapted for high-quality translation between Ekegusii, Kiswahili, and English using limited multilingual resources?*

---

## 🏛️ Repository Architecture

```text
Ekegusii-LLM-Translation/ (transaltion_model)
│
├── ⚙️ configs/                          # Reproducible YAML Configurations
│   ├── models/                          (aya_8b.yaml, llama31_8b.yaml, common.yaml)
│   ├── training/                        (qlora.yaml, optimizer.yaml, scheduler.yaml)
│   ├── datasets/                        (master.yaml, monolingual.yaml, bilingual.yaml)
│   └── prompts/                         (templates.yaml, translation.yaml, lexical.yaml)
│
├── 📊 data/                             # Master Corpus & Dataset Database
│   ├── raw/                             (monolingual/, bilingual/, trilingual/, dictionaries/)
│   ├── master_corpus/
│   │   ├── master_sentence_corpus.csv   (49,277 Multilingual Concepts)
│   │   ├── master_lexical_corpus.csv    (268 Lexicon Dictionary Term Entries)
│   │   └── splits/                      (Strict 80/10/10 Zero-Leakage Master Splits)
│   ├── processed/
│   ├── augmented/
│   └── cache/
│
├── 📓 notebooks/                         # 13 Numbered Reproducible Research Notebooks
│   ├── 01_master_corpus_analysis.ipynb  (01. Master Corpus & Domain Analysis)
│   ├── 02_data_validation.ipynb        (02. Data Validation & Leakage Verification)
│   ├── 03_resource_statistics.ipynb    (03. Resource Statistics & Coverage)
│   ├── 04_tokenizer_analysis.ipynb     (04. Aya 23 vs Llama 3.1 Subword Fertility)
│   ├── 05_instruction_generation.ipynb (05. 6-Way Multilingual Instruction Generator)
│   ├── 06_dataset_scheduler.ipynb      (06. Dynamic Dataset Scheduler & Sampler)
│   ├── 07_train_aya.ipynb              (07. Cohere Aya-23 8B QLoRA Fine-Tuning Engine)
│   ├── 08_train_llama.ipynb            (08. Meta Llama-3.1 8B QLoRA Fine-Tuning Engine)
│   ├── 09_translation_evaluation.ipynb (09. SacreBLEU, chrF++, & COMET Evaluation)
│   ├── 10_dictionary_analysis.ipynb    (10. Lexical Term Accuracy & Rare-Word Study)
│   ├── 11_ablation_study.ipynb         (11. Resource Attribution Ablation E0 - E8)
│   ├── 12_error_analysis.ipynb         (12. Qualitative Translation Error Analysis)
│   └── 13_publication_figures.ipynb    (13. Publication-Ready Figures & Dashboards)
│
├── 🐍 src/                               # Modular Python Codebase
│   ├── master_corpus/                   (manager.py, integrity.py, provenance.py, scheduler.py)
│   ├── preprocessing/                   (normalize.py, deduplicate.py, filtering.py)
│   ├── tokenizer/                       (aya.py, llama.py, metrics.py, compare.py)
│   ├── task_generation/                 (instruction_generator.py, prompt_templates.py)
│   ├── datasets/                        (builder.py, dataloader.py, collator.py)
│   ├── models/                          (aya/ and llama/ QLoRA trainers & inference)
│   ├── evaluation/                      (sacrebleu.py, chrf.py, lexical_accuracy.py)
│   ├── experiments/                     (ablation.py, baseline.py, curriculum.py)
│   ├── visualization/                   (publication.py, dashboards.py)
│   ├── utils/                           (config.py, logger.py, seed.py, checkpoint.py)
│   └── cli/                             (train.py, evaluate.py, translate.py, analyze.py)
│
├── 🔬 experiments/                      # Experiment Output Logs & Checkpoints (E0 to E8)
├── 💾 checkpoints/                      # Fine-Tuned Model Weights (Aya 8B & Llama 3.1 8B)
├── 📈 outputs/                          # Predictions, Metrics, Tables, and Figures
├── 📜 scripts/                          # Shell scripts for 1-command pipeline execution
├── 🧪 tests/                            # Unit test suite for data loaders & metrics
├── 📚 docs/                             # Architecture, Methodology, and API Documentation
└── 📝 paper/                            # LaTeX Manuscript, Tables, Figures, and Appendix
```

---

## 📊 Master Corpus Statistics (0% Data Leakage Guarantee)

* **Master Sentence Corpus**: **49,277 multilingual concepts** (`concept_id`, `English`, `Kiswahili`, `Ekegusii`, `source`).
* **Master Lexical Corpus**: **268 dictionary terms** (Isolated for rare-word precision evaluation).
* **Master 80/10/10 Split**:
  - **Train Split**: 39,421 concepts (80%)
  - **Validation Split**: 4,928 concepts (10%)
  - **Test Split**: 4,928 concepts (10%) — **0 overlapping concept IDs across splits**.
* **6-Way Instruction Expansion**: Generates **234,650 supervised instruction-tuning tasks** (`English` ↔ `Ekegusii` ↔ `Kiswahili`).

---

## 🧪 Systematic Experiments Matrix (E0 to E8)

| Experiment | ENG ↔ EKE | SWA ↔ EKE | ENG ↔ SWA | Trilingual | Dictionary | Goal / Target Hypothesis Tested |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **E0** | ✗ | ✗ | ✗ | ✗ | ✗ | Base Model Zero-Shot Baseline |
| **E1** | ✓ | ✗ | ✗ | ✗ | ✗ | Direct Bilingual Translation |
| **E2** | ✗ | ✓ | ✗ | ✗ | ✗ | Swahili Cross-Lingual Transfer |
| **E3** | ✓ | ✓ | ✗ | ✗ | ✗ | Combined Dual Bilingual |
| **E4** | ✓ | ✓ | ✓ | ✓ | ✗ | **H2**: Multilingual Supervision Boost |
| **E5** | ✓ | ✓ | ✓ | ✓ | ✗ | **H1**: Full Sentence Corpus Exposure |
| **E6** | ✓ | ✓ | ✓ | ✓ | ✓ | **H3**: Lexical Dictionary Augmentation |
| **E7** | ✓ | ✓ | ✓ | ✓ | ✓ | Curriculum Resource Scheduling |
| **E8** | ✓ | ✓ | ✓ | ✓ | ✓ | Final Production Ensemble System |

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
git clone https://github.com/aykahsay/transaltion_model.git
cd transaltion_model
pip install -r requirements.txt
```

### 2. Verify Master Corpus Integrity (0% Data Leakage Test)
```bash
python -m src.master_corpus.integrity
```

### 3. Generate 6-Way Instruction Tuning Task Datasets
```bash
python -m src.data_processing.instruction_task_generator
```

### 4. Run Resource Attribution Analysis
```bash
python -m src.evaluation.resource_attribution_analyzer
```

---

## 📄 Citation

If you use this repository, master corpus, or instruction-tuning framework in your research, please cite:

```bibtex
@article{ekegusii_llm_nmt_2026,
  title={Resource-Aware Adaptation of Multilingual Large Language Models for Low-Resource Machine Translation: A Case Study on Ekegusii},
  author={Aykahsay et al.},
  journal={ACL / EMNLP Research Preprints},
  year={2026},
  url={https://github.com/aykahsay/transaltion_model}
}
```

---
*Maintained by Aykahsay Research Team (2026)*
