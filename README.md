# 🌍 Resource-Aware Adaptation of Multilingual Large Language Models for Low-Resource Machine Translation: A Case Study on Ekegusii

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-PEFT-orange.svg)](https://huggingface.co/)

Official open-source repository for the research paper:
> **"Resource-Aware Adaptation of Multilingual Large Language Models for Low-Resource Machine Translation: A Case Study on Ekegusii"**

---

## 🏛️ Repository Architecture
```text
Ekegusii-LLM-Translation/
├── configs/             # Training, Dataset, and Generation YAML configurations
├── data/
│   ├── raw/             # Raw monolingual, bilingual, trilingual datasets
│   └── master_corpus/   # 49,277 Multilingual Concepts with 0% leakage splits
├── notebooks/           # 10 Numbered reproducible Jupyter notebooks (01 to 10)
├── src/                 # Modular Python codebase (master_corpus, models, eval, etc.)
├── experiments/         # Output logs for Experiments E0 through E6
└── paper/               # Manuscript, tables, and publication-ready figures
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Verify Master Corpus & Data Leakage
```bash
python -m src.master_corpus.integrity
```

### 3. Generate 6-Way Instruction Tuning Datasets
```bash
python -m src.data_processing.instruction_task_generator
```

---

## 📄 Citation
```bibtex
@article{ekegusii_llm_nmt_2026,
  title={Resource-Aware Adaptation of Multilingual Large Language Models for Low-Resource Machine Translation: A Case Study on Ekegusii},
  author={Aykahsay et al.},
  journal={ACL / EMNLP Research Preprints},
  year={2026}
}
```
