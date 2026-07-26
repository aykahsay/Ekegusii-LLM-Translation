"""
build_colab_chunks.py
---------------------
Generates notebooks/colab_training.ipynb with self-contained, well-commented,
step-by-step Python code chunks for easy execution and understanding in Google Colab.
"""

import json, os

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": { "id": "view-in-github" },
      "source": [
        "<a href=\"https://colab.research.google.com/github/aykahsay/Multilogual_transaltion_nlp/blob/main/notebooks/colab_training.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
        "# 🇰🇪 Multilingual Public Service Announcement (PSA) Translation Studio\n",
        "### Fine-Tuning NMT Models for English ↔ Swahili ↔ Ekegusii (Gusii)\n",
        "\n",
        "This interactive notebook guides you step-by-step through dataset loading, preprocessing, model fine-tuning (MarianMT & Meta NLLB-200), automatic evaluation (BLEU / SacreBLEU / chrF), and batch inference.\n",
        "\n",
        "**⚡ Prerequisites:** Go to **Runtime > Change runtime type** and select **T4 GPU** before running!"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🛠️ Step 1: Environment Setup & Repository Cloning"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "# 1. Clone repository from GitHub\n",
        "!git clone https://github.com/aykahsay/Multilogual_transaltion_nlp.git\n",
        "%cd Multilogual_transaltion_nlp\n",
        "\n",
        "# 2. Install required packages\n",
        "!pip install -q transformers datasets evaluate sacrebleu sentencepiece sacremoses torch accelerate pandas scikit-learn tqdm\n",
        "\n",
        "# 3. Verify GPU availability\n",
        "import torch\n",
        "print(f\"PyTorch Version: {torch.__version__}\")\n",
        "print(f\"GPU Available : {torch.cuda.is_available()}\")\n",
        "if torch.cuda.is_available():\n",
        "    print(f\"GPU Model     : {torch.cuda.get_device_name(0)}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📊 Step 2: Dataset Loading & Language Splitting\n",
        "Loads `data/Master_PSA_Only.csv` and splits it into language-specific parallel corpora:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "import pandas as pd\n",
        "\n",
        "# Path configuration\n",
        "master_path = \"data/Master_PSA_Only.csv\"\n",
        "pending_marker = \"N/A - Pending Fine-Tuned Model Inference\"\n",
        "\n",
        "df = pd.read_csv(master_path, dtype=str)\n",
        "print(f\"Loaded {len(df):,} total rows from {master_path}\")\n",
        "\n",
        "# Filter valid parallel pairs\n",
        "has_swa = df['Kiswahili'].fillna('').str.strip().ne('') & (df['Kiswahili'] != pending_marker)\n",
        "has_guz = df['Ekegusii'].fillna('').str.strip().ne('') & (df['Ekegusii'] != pending_marker)\n",
        "\n",
        "en_sw_df = df[has_swa][['English', 'Kiswahili', 'Domain']].reset_index(drop=True)\n",
        "en_guz_df = df[has_guz][['English', 'Ekegusii', 'Domain']].reset_index(drop=True)\n",
        "trilingual_df = df[has_swa & has_guz][['English', 'Kiswahili', 'Ekegusii', 'Domain']].reset_index(drop=True)\n",
        "\n",
        "print(f\"✅ English - Swahili Parallel Pairs : {len(en_sw_df):,}\")\n",
        "print(f\"✅ English - Ekegusii Parallel Pairs: {len(en_guz_df):,}\")\n",
        "print(f\"✅ Complete Trilingual Triplets    : {len(trilingual_df):,}\")\n",
        "\n",
        "print(\"\\nSample English-Ekegusii Pair:\")
        "print(en_guz_df.head(2).to_dict(orient='records'))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🚀 Step 3: Fine-Tuning Meta NLLB-200 (English ➡️ Ekegusii)\n",
        "Fine-tunes `facebook/nllb-200-distilled-600M` on English-Ekegusii parallel sentences (`eng_Latn` -> `guz_Latn`)."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import torch\n",
        "from transformers import (\n",
        "    AutoTokenizer, \n",
        "    AutoModelForSeq2SeqLM, \n",
        "    DataCollatorForSeq2Seq, \n",
        "    Seq2SeqTrainingArguments, \n",
        "    Seq2SeqTrainer\n",
        ")\n",
        "from datasets import Dataset\n",
        "\n",
        "model_checkpoint = \"facebook/nllb-200-distilled-600M\"\n",
        "output_dir = \"models/nllb-en-guz\"\n",
        "os.makedirs(output_dir, exist_ok=True)\n",
        "\n",
        "print(f\"Loading tokenizer & model: {model_checkpoint}...\")\n",
        "tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang=\"eng_Latn\", tgt_lang=\"guz_Latn\")\n",
        "model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)\n",
        "\n",
        "# Prepare Train / Validation Split (90% train, 10% val)\n",
        "shuffled_guz = en_guz_df.sample(frac=1, random_state=42).reset_index(drop=True)\n",
        "split_idx = int(0.9 * len(shuffled_guz))\n",
        "\n",
        "train_data = Dataset.from_pandas(shuffled_guz.iloc[:split_idx])\n",
        "val_data = Dataset.from_pandas(shuffled_guz.iloc[split_idx:])\n",
        "\n",
        "def preprocess_nllb(examples):\n",
        "    inputs = [str(ex) for ex in examples[\"English\"]]\n",
        "    targets = [str(ex) for ex in examples[\"Ekegusii\"]]\n",
        "    model_inputs = tokenizer(inputs, max_length=128, truncation=True)\n",
        "    labels = tokenizer(text_target=targets, max_length=128, truncation=True)\n",
        "    model_inputs[\"labels\"] = labels[\"input_ids\"]\n",
        "    return model_inputs\n",
        "\n",
        "print(\"Tokenizing Ekegusii datasets...\")\n",
        "tokenized_train = train_data.map(preprocess_nllb, batched=True, remove_columns=train_data.column_names)\n",
        "tokenized_val = val_data.map(preprocess_nllb, batched=True, remove_columns=val_data.column_names)\n",
        "\n",
        "data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)\n",
        "\n",
        "training_args = Seq2SeqTrainingArguments(\n",
        "    output_dir=output_dir,\n",
        "    eval_strategy=\"epoch\",\n",
        "    learning_rate=5e-5,\n",
        "    per_device_train_batch_size=8,\n",
        "    per_device_eval_batch_size=8,\n",
        "    weight_decay=0.01,\n",
        "    save_total_limit=2,\n",
        "    num_train_epochs=3,\n",
        "    predict_with_generate=True,\n",
        "    fp16=torch.cuda.is_available(),\n",
        "    logging_steps=50,\n",
        "    save_strategy=\"epoch\",\n",
        "    report_to=\"none\"\n",
        ")\n",
        "\n",
        "trainer = Seq2SeqTrainer(\n",
        "    model=model,\n",
        "    args=training_args,\n",
        "    train_dataset=tokenized_train,\n",
        "    eval_dataset=tokenized_val,\n",
        "    processing_class=tokenizer,\n",
        "    data_collator=data_collator,\n",
        ")\n",
        "\n",
        "print(\"\\n🔥 Starting NLLB-200 Ekegusii Training...\")\n",
        "trainer.train()\n",
        "\n",
        "# Save fine-tuned model\n",
        "trainer.save_model(output_dir)\n",
        "tokenizer.save_pretrained(output_dir)\n",
        "print(f\"✅ Ekegusii Model saved to {output_dir}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🌍 Step 4: Fine-Tuning MarianMT (English ➡️ Swahili)\n",
        "Fine-tunes `Helsinki-NLP/opus-mt-en-sw` on English-Swahili parallel sentences."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from transformers import MarianTokenizer, AutoModelForSeq2SeqLM\n",
        "\n",
        "sw_model_checkpoint = \"Helsinki-NLP/opus-mt-en-sw\"\n",
        "sw_output_dir = \"models/psa-en-sw-finetuned\"\n",
        "os.makedirs(sw_output_dir, exist_ok=True)\n",
        "\n",
        "print(f\"Loading tokenizer & model: {sw_model_checkpoint}...\")\n",
        "sw_tokenizer = MarianTokenizer.from_pretrained(sw_model_checkpoint)\n",
        "sw_model = AutoModelForSeq2SeqLM.from_pretrained(sw_model_checkpoint)\n",
        "\n",
        "# Prepare Train / Validation Split (90% train, 10% val)\n",
        "shuffled_sw = en_sw_df.sample(frac=1, random_state=42).reset_index(drop=True)\n",
        "sw_split_idx = int(0.9 * len(shuffled_sw))\n",
        "\n",
        "sw_train_data = Dataset.from_pandas(shuffled_sw.iloc[:sw_split_idx])\n",
        "sw_val_data = Dataset.from_pandas(shuffled_sw.iloc[sw_split_idx:])\n",
        "\n",
        "def preprocess_marian(examples):\n",
        "    inputs = [str(ex) for ex in examples[\"English\"]]\n",
        "    targets = [str(ex) for ex in examples[\"Kiswahili\"]]\n",
        "    model_inputs = sw_tokenizer(inputs, text_target=targets, max_length=128, padding=\"max_length\", truncation=True)\n",
        "    model_inputs[\"labels\"] = [\n",
        "        [(l if l != sw_tokenizer.pad_token_id else -100) for l in label] for label in model_inputs[\"labels\"]\n",
        "    ]\n",
        "    return model_inputs\n",
        "\n",
        "print(\"Tokenizing Swahili datasets...\")\n",
        "sw_tokenized_train = sw_train_data.map(preprocess_marian, batched=True, remove_columns=sw_train_data.column_names)\n",
        "sw_tokenized_val = sw_val_data.map(preprocess_marian, batched=True, remove_columns=sw_val_data.column_names)\n",
        "\n",
        "sw_collator = DataCollatorForSeq2Seq(sw_tokenizer, model=sw_model)\n",
        "\n",
        "sw_training_args = Seq2SeqTrainingArguments(\n",
        "    output_dir=sw_output_dir,\n",
        "    eval_strategy=\"epoch\",\n",
        "    learning_rate=2e-5,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=16,\n",
        "    weight_decay=0.01,\n",
        "    save_total_limit=2,\n",
        "    num_train_epochs=3,\n",
        "    predict_with_generate=True,\n",
        "    fp16=torch.cuda.is_available(),\n",
        "    logging_steps=50,\n",
        "    save_strategy=\"epoch\",\n",
        "    report_to=\"none\"\n",
        ")\n",
        "\n",
        "sw_trainer = Seq2SeqTrainer(\n",
        "    model=sw_model,\n",
        "    args=sw_training_args,\n",
        "    train_dataset=sw_tokenized_train,\n",
        "    eval_dataset=sw_tokenized_val,\n",
        "    processing_class=sw_tokenizer,\n",
        "    data_collator=sw_collator,\n",
        ")\n",
        "\n",
        "print(\"\\n🔥 Starting Swahili MarianMT Training...\")\n",
        "sw_trainer.train()\n",
        "\n",
        "# Save fine-tuned model\n",
        "sw_trainer.save_model(sw_output_dir)\n",
        "sw_tokenizer.save_pretrained(sw_output_dir)\n",
        "print(f\"✅ Swahili Model saved to {sw_output_dir}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📈 Step 5: Automatic Model Evaluation (`BLEU` & `SacreBLEU` & `chrF`)\n",
        "Computes quantitative evaluation metrics comparing model predictions against ground truth target sentences."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import evaluate\n",
        "from tqdm import tqdm\n",
        "\n",
        "print(\"Loading evaluation metrics...\")\n",
        "sacrebleu_metric = evaluate.load(\"sacrebleu\")\n",
        "chrf_metric = evaluate.load(\"chrf\")\n",
        "\n",
        "# Evaluate 100 validation samples\n",
        "eval_sample = val_data.select(range(min(100, len(val_data))))\n",
        "references = [[str(ex)] for ex in eval_sample[\"Ekegusii\"]]\n",
        "inputs = [str(ex) for ex in eval_sample[\"English\"]]\n",
        "\n",
        "predictions = []
        "print(\"Generating Ekegusii translation predictions...\")\n",
        "for text in tqdm(inputs):\n",
        "    input_ids = tokenizer(text, return_tensors=\"pt\").input_ids.to(model.device)\n",
        "    outputs = model.generate(input_ids, max_length=128)\n",
        "    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)\n",
        "    predictions.append(pred_text)\n",
        "\n",
        "bleu_res = sacrebleu_metric.compute(predictions=predictions, references=references)\n",
        "chrf_res = chrf_metric.compute(predictions=predictions, references=references)\n",
        "\n",
        "print(\"\\n========================================\")\n",
        "print(\"  AUTOMATIC TRANSLATION EVALUATION RESULTS\")\n",
        "print(\"========================================\")\n",
        "print(f\"  • SacreBLEU Score : {bleu_res['score']:.2f}\")\n",
        "print(f\"  • chrF Score      : {chrf_res['score']:.2f}\")\n",
        "print(\"========================================\")\n",
        "\n",
        "print(\"\\nSample Translation Pair:\")\n",
        "print(f\"  Source (EN) : {inputs[0]}\")\n",
        "print(f\"  Ref (GUZ)   : {references[0][0]}\")\n",
        "print(f\"  Pred (GUZ)  : {predictions[0]}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🧪 Step 6: Interactive Single Sentence Translation Inference"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "def translate_psa(english_text):\n",
        "    # 1. Swahili Translation\n",
        "    sw_inputs = sw_tokenizer(english_text, return_tensors=\"pt\").input_ids.to(sw_model.device)\n",
        "    sw_outs = sw_model.generate(sw_inputs, max_length=128)\n",
        "    sw_text = sw_tokenizer.decode(sw_outs[0], skip_special_tokens=True)\n",
        "    \n",
        "    # 2. Ekegusii Translation\n",
        "    guz_inputs = tokenizer(english_text, return_tensors=\"pt\").input_ids.to(model.device)\n",
        "    guz_outs = model.generate(guz_inputs, max_length=128)\n",
        "    guz_text = tokenizer.decode(guz_outs[0], skip_special_tokens=True)\n",
        "    \n",
        "    return {\n",
        "        \"English\": english_text,\n",
        "        \"Kiswahili\": sw_text,\n",
        "        \"Ekegusii\": guz_text\n",
        "    }\n",
        "\n",
        "sample_input = \"Wash your hands frequently with soap and running water to prevent the spread of diseases.\"\n",
        "res = translate_psa(sample_input)\n",
        "\n",
        "print(\"📢 TRANSLATION RESULT:\")\n",
        "print(f\"  [EN] : {res['English']}\")\n",
        "print(f\"  [SW] : {res['Kiswahili']}\")\n",
        "print(f\"  [GUZ]: {res['Ekegusii']}\")"
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

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/colab_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated notebooks/colab_training.ipynb with self-contained python code chunks successfully!")
