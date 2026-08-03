"""
Professional Open-Source Research Repository Builder
------------------------------------------------------
Constructs the complete ACL/EMNLP production repository structure for 
Ekegusii-LLM-Translation (Aya 8B / Llama 3.1 8B QLoRA Translation Framework).
"""

import os
import shutil
import json
import yaml
import pandas as pd

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"

def create_directory_structure():
    print("=== Step 1: Building Directory Tree ===")
    
    dirs = [
        "configs/training", "configs/datasets", "configs/generation",
        "data/raw/monolingual", "data/raw/bilingual", "data/raw/trilingual", "data/raw/dictionaries",
        "data/processed", "data/cache",
        "data/master_corpus/splits",
        "notebooks",
        "src/master_corpus",
        "src/data",
        "src/preprocessing",
        "src/tokenizer",
        "src/tasks",
        "src/models/aya", "src/models/llama",
        "src/evaluation",
        "src/experiments",
        "src/visualization",
        "src/utils",
        "src/cli",
        "experiments/E0_baseline", "experiments/E1_monolingual", "experiments/E2_bilingual",
        "experiments/E3_trilingual", "experiments/E4_full_resources", "experiments/E5_dictionary", "experiments/E6_curriculum",
        "checkpoints/aya", "checkpoints/llama",
        "outputs/predictions", "outputs/metrics", "outputs/logs", "outputs/tables", "outputs/figures",
        "scripts", "tests", "docs", "paper/tables", "paper/figures", "paper/supplementary", "paper/manuscript"
    ]
    
    for d in dirs:
        full_path = os.path.join(WORKSPACE_DIR, d)
        os.makedirs(full_path, exist_ok=True)
        # Add .gitkeep to ensure empty folders are tracked by Git
        gitkeep = os.path.join(full_path, ".gitkeep")
        if not os.path.exists(gitkeep):
            with open(gitkeep, 'w') as f:
                f.write('')
                
    print("[OK] Directory Tree Built Successfully!")

def populate_configs():
    print("=== Step 2: Populating Configuration YAML Files ===")
    
    # 1. Training Configs
    aya_config = {
        'model': {
            'name_or_path': 'CohereForAI/aya-23-8B',
            'torch_dtype': 'bfloat16',
            'use_fast_tokenizer': True
        },
        'qlora': {
            'r': 32,
            'lora_alpha': 64,
            'lora_dropout': 0.05,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        },
        'training': {
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2.0e-4,
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.1,
            'num_train_epochs': 4,
            'optim': 'paged_adamw_8bit'
        }
    }
    
    llama_config = {
        'model': {
            'name_or_path': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'torch_dtype': 'bfloat16'
        },
        'qlora': {
            'r': 32,
            'lora_alpha': 64,
            'lora_dropout': 0.05,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
        },
        'training': {
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2.0e-4,
            'lr_scheduler_type': 'cosine',
            'warmup_ratio': 0.1,
            'num_train_epochs': 4
        }
    }
    
    with open(os.path.join(WORKSPACE_DIR, "configs/training/aya_8b_qlora.yaml"), 'w') as f:
        yaml.dump(aya_config, f)
    with open(os.path.join(WORKSPACE_DIR, "configs/training/llama31_8b_qlora.yaml"), 'w') as f:
        yaml.dump(llama_config, f)
        
    print("[OK] YAML Configurations Saved!")

def populate_master_corpus_module():
    print("=== Step 3: Populating Master Corpus Manager Module ===")
    
    manager_code = '''"""
Master Corpus Manager API
-------------------------
Centralized API for accessing master sentence and lexical corpora with 0% leakage guarantee.
"""

import os
import pandas as pd

class MasterCorpusManager:
    def __init__(self, data_dir="data/master_corpus"):
        self.data_dir = data_dir
        self.sentence_corpus_path = os.path.join(data_dir, "master_sentence_corpus.csv")
        self.lexical_corpus_path = os.path.join(data_dir, "master_lexical_corpus.csv")
        self.splits_dir = os.path.join(data_dir, "splits")
        
    def get_train_split(self):
        return pd.read_csv(os.path.join(self.splits_dir, "master_train.csv"))
        
    def get_val_split(self):
        return pd.read_csv(os.path.join(self.splits_dir, "master_val.csv"))
        
    def get_test_split(self):
        return pd.read_csv(os.path.join(self.splits_dir, "master_test.csv"))
        
    def verify_integrity(self):
        train = self.get_train_split()
        val = self.get_val_split()
        test = self.get_test_split()
        
        train_ids = set(train['concept_id'])
        val_ids = set(val['concept_id'])
        test_ids = set(test['concept_id'])
        
        overlap = (train_ids & test_ids) | (train_ids & val_ids) | (val_ids & test_ids)
        assert len(overlap) == 0, f"Data leakage detected! Overlap: {len(overlap)}"
        return True
'''
    with open(os.path.join(WORKSPACE_DIR, "src/master_corpus/manager.py"), 'w') as f:
        f.write(manager_code)
        
    integrity_code = '''"""
Data Leakage & Integrity Checker
"""
from src.master_corpus.manager import MasterCorpusManager

if __name__ == "__main__":
    manager = MasterCorpusManager()
    if manager.verify_integrity():
        print("[SUCCESS] Master Corpus 0% Data Leakage Verified!")
'''
    with open(os.path.join(WORKSPACE_DIR, "src/master_corpus/integrity.py"), 'w') as f:
        f.write(integrity_code)

def generate_numbered_notebooks():
    print("=== Step 4: Generating Numbered Research Notebooks (01 to 10) ===")
    
    notebook_titles = [
        ("01_master_corpus_analysis.ipynb", "01. Master Corpus Distribution & Domain Analysis"),
        ("02_data_quality.ipynb", "02. Data Quality & Preprocessing Statistics"),
        ("03_tokenizer_analysis.ipynb", "03. Aya vs Llama Subword Tokenizer Fertility Analysis"),
        ("04_dataset_generation.ipynb", "04. 6-Way Multilingual Instruction Dataset Generation"),
        ("05_translation_baseline.ipynb", "05. Baseline Machine Translation Evaluation"),
        ("06_aya_training.ipynb", "06. Cohere Aya-23 8B QLoRA Fine-Tuning"),
        ("07_llama_training.ipynb", "07. Meta Llama-3.1 8B QLoRA Fine-Tuning"),
        ("08_evaluation.ipynb", "08. Comprehensive SacreBLEU, chrF++, & COMET Evaluation"),
        ("09_resource_ablation.ipynb", "09. Resource Contribution & Ablation Study (E0 - E6)"),
        ("10_visualizations.ipynb", "10. Publication Figures & Learning Curves")
    ]
    
    for filename, title in notebook_titles:
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# 🔬 {title}\n", f"**Ekegusii-LLM-Translation Project Notebook**"]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import torch\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "print('PyTorch Version:', torch.__version__)\n",
                        "print('CUDA Available:', torch.cuda.is_available())"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.12"}
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        
        target_path = os.path.join(WORKSPACE_DIR, "notebooks", filename)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=2)
            
    print("[OK] 10 Numbered Research Notebooks Generated!")

def create_root_project_files():
    print("=== Step 5: Generating Root Open-Source Metadata Files ===")
    
    readme_content = """# 🌍 Resource-Aware Adaptation of Multilingual Large Language Models for Low-Resource Machine Translation: A Case Study on Ekegusii

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
"""
    with open(os.path.join(WORKSPACE_DIR, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme_content)
        
    pyproject_content = """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "ekegusii-llm-translation"
version = "1.0.0"
description = "Resource-Aware Adaptation of Multilingual LLMs for Low-Resource Machine Translation"
authors = [{ name = "Aykahsay" }]
license = { text = "MIT" }
requires-python = ">=3.10"
"""
    with open(os.path.join(WORKSPACE_DIR, "pyproject.toml"), 'w', encoding='utf-8') as f:
        f.write(pyproject_content)
        
    print("[OK] Root README.md and pyproject.toml Generated!")

if __name__ == "__main__":
    create_directory_structure()
    populate_configs()
    populate_master_corpus_module()
    generate_numbered_notebooks()
    create_root_project_files()
    print("\n[SUCCESS] Production Open-Source Research Repository Built!")
