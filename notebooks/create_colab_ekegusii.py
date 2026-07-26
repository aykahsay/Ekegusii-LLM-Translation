"""
create_colab_ekegusii.py
------------------------
Generates train_ekegusii_colab.ipynb configured to clone GitHub,
install dependencies, and execute Ekegusii NLLB fine-tuning.
"""

import json

colab_notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 🇰🇪 Fine-Tuning NLLB-200 for English ➡️ Ekegusii (Gusii) Machine Translation\n",
        "This notebook trains a Meta NLLB-200 neural machine translation model on English to Ekegusii parallel sentences."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# 1. Clone GitHub repository & check GPU\n",
        "!git clone https://github.com/aykahsay/Multilogual_transaltion_nlp.git\n",
        "%cd Multilogual_transaltion_nlp\n",
        "!nvidia-smi"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# 2. Install required packages\n",
        "!pip install -q transformers datasets evaluate sacrebleu sentencepiece torch accelerate"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# 3. Run dataset language split\n",
        "!python src/data_processing/split_by_language.py"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# 4. Run Ekegusii fine-tuning\n",
        "!python src/training_eval/train_ekegusii.py"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 2
}

with open("notebooks/train_ekegusii_colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(colab_notebook, f, indent=2)

print("Updated notebooks/train_ekegusii_colab.ipynb successfully!")
