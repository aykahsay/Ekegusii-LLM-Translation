"""
Builder for colab_clone_and_run.ipynb Notebook
----------------------------------------------
Constructs a single-click Colab notebook that clones the official GitHub repository 
https://github.com/aykahsay/Ekegusii-LLM-Translation.git, installs dependencies, 
runs data integrity tests, generates tasks, and executes the entire NMT training pipeline.
"""

import os
import json

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"

def create_colab_clone_notebook():
    print("=== Constructing colab_clone_and_run.ipynb Notebook ===")
    
    cells = [
        # Cell 1: Markdown Header
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# ☁️ One-Click Colab Runner: `colab_clone_and_run.ipynb`\n",
                "### Resource-Aware Adaptation of Multilingual LLMs for Ekegusii Machine Translation\n",
                "**Official GitHub Repo**: [`https://github.com/aykahsay/Ekegusii-LLM-Translation.git`](https://github.com/aykahsay/Ekegusii-LLM-Translation.git)\n\n",
                "---"
            ]
        },
        
        # Cell 2: Git Clone Repository
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📥 Step 1: Clone Official GitHub Repository"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Clone the lightweight GitHub repository\n",
                "!git clone https://github.com/aykahsay/Ekegusii-LLM-Translation.git\n\n",
                "# 2. Change working directory into the project folder\n",
                "%cd Ekegusii-LLM-Translation\n",
                "!pwd"
            ]
        },
        
        # Cell 3: Install Dependencies
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📦 Step 2: Install Project Requirements"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -r requirements.txt"
            ]
        },
        
        # Cell 4: Hardware Check & VRAM Reset
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ Step 3: GPU Hardware Check & Memory Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "import gc\n\n",
                "print('=== GPU HARDWARE CONFIGURATION ===')\n",
                "print('PyTorch Version:', torch.__version__)\n",
                "print('CUDA Available:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('GPU Device Name:', torch.cuda.get_device_name(0))\n",
                "    print('Total GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')\n\n",
                "gc.collect()\n",
                "if torch.cuda.is_available():\n",
                "    torch.cuda.empty_cache()\n",
                "print('[OK] GPU Memory Cleaned & Ready!')"
            ]
        },
        
        # Cell 5: Verify Data Leakage
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 Step 4: Verify Master Corpus 0% Data Leakage"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python -m src.master_corpus.integrity"
            ]
        },
        
        # Cell 6: Generate Instruction Tasks
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🔄 Step 5: Generate 6-Way Multilingual Instruction Tasks"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python -m src.task_generation.instruction_generator"
            ]
        },
        
        # Cell 7: Execute Master Training & Evaluation Pipeline
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🚀 Step 6: Execute Full NMT Training & Evaluation Pipeline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run the master execution notebook\n",
                "%run run_all.ipynb"
            ]
        },
        
        # Cell 8: Resource Attribution Report
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📈 Step 7: Compute Resource Attribution Matrix & Report"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python -m src.evaluation.resource_attribution_analyzer"
            ]
        }
    ]
    
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # Save both in root directory and in notebooks/ directory
    with open(os.path.join(WORKSPACE_DIR, "colab_clone_and_run.ipynb"), 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
        
    with open(os.path.join(WORKSPACE_DIR, "notebooks", "colab_clone_and_run.ipynb"), 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
        
    print("[OK] colab_clone_and_run.ipynb Created in Root and notebooks/!")

if __name__ == "__main__":
    create_colab_clone_notebook()
