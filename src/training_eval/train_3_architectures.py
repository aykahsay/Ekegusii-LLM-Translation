import os
import re
import sys
import glob
import pandas as pd
import numpy as np
import torch
import transformers
import peft
import datasets
import evaluate
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, TaskType

print("=== NVIDIA A100 GPU Master Training Pipeline: 3 NMT Architectures ===")
print("PyTorch Version:", torch.__version__)
print("Transformers Version:", transformers.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("Total VRAM:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANG_TAGS = {"English": "eng_Latn", "Kiswahili": "swh_Latn", "Ekegusii": "swh_Latn"}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

def find_data_file(filename):
    possible_paths = [
        os.path.join("..", "data", "data_trian_tringual", filename),
        os.path.join("..", "data", "data_train_bilingual", filename),
        os.path.join("..", "data", "data_train_unilingual", filename),
        os.path.join("data", "data_trian_tringual", filename),
        os.path.join("data", "data_train_bilingual", filename),
        os.path.join("data", "data_train_unilingual", filename),
        os.path.join("data_trian_tringual", filename),
        os.path.join("data_train_bilingual", filename),
        os.path.join("data_train_unilingual", filename),
        filename,
        os.path.join("..", filename)
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    matches = glob.glob(f"**/{filename}", recursive=True) + glob.glob(f"../**/{filename}", recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate dataset file '{filename}' in workspace.")

def preprocess_nmt_function(examples, src_lang="English", tgt_lang="Ekegusii"):
    inputs = [str(x) for x in examples[src_lang]]
    targets = [str(x) for x in examples[tgt_lang]]
    tokenizer.src_lang = LANG_TAGS.get(src_lang, "eng_Latn")
    tokenizer.tgt_lang = LANG_TAGS.get(tgt_lang, "swh_Latn")
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=False)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    chrf = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": bleu["score"], "chrf": chrf["score"]}

def evaluate_on_test(model, test_df, src_lang="English", tgt_lang="Ekegusii"):
    test_ds = Dataset.from_pandas(test_df).map(lambda x: preprocess_nmt_function(x, src_lang, tgt_lang), batched=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=32,
        predict_with_generate=True,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none"
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics
    )
    results = trainer.evaluate(test_ds)
    return results.get("eval_bleu", 0.0), results.get("eval_chrf", 0.0)

def train_and_evaluate():
    splits_root = "data_splits"
    arch1_dir = os.path.join(splits_root, "arch1_direct_trilingual_psa")
    
    a1_train = pd.read_csv(os.path.join(arch1_dir, "train.csv"))
    a1_val = pd.read_csv(os.path.join(arch1_dir, "val.csv"))
    a1_test = pd.read_csv(os.path.join(arch1_dir, "test.csv"))
    
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.1)
    
    # --- ARCHITECTURE 1 ---
    print("\n=== RUNNING ARCHITECTURE 1: Direct Trilingual Fine-Tuning ===")
    base_model1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    a1_model = get_peft_model(base_model1, peft_config)
    
    train_ds1 = Dataset.from_pandas(a1_train).map(lambda x: preprocess_nmt_function(x, "English", "Ekegusii"), batched=True)
    val_ds1 = Dataset.from_pandas(a1_val).map(lambda x: preprocess_nmt_function(x, "English", "Ekegusii"), batched=True)
    
    args1 = Seq2SeqTrainingArguments(
        output_dir="./output_arch1",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        predict_with_generate=True,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=20,
        report_to="none"
    )
    trainer1 = Seq2SeqTrainer(
        model=a1_model,
        args=args1,
        train_dataset=train_ds1,
        eval_dataset=val_ds1,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=a1_model),
        compute_metrics=compute_metrics
    )
    trainer1.train()
    bleu1, chrf1 = evaluate_on_test(a1_model, a1_test)
    print(f"🏆 Architecture 1 Test Results: SacreBLEU = {bleu1:.2f} | chrF = {chrf1:.2f}")
    
    # --- ARCHITECTURE 2 ---
    print("\n=== RUNNING ARCHITECTURE 2: Sequential Transfer Learning ===")
    a2_bi = pd.read_csv(find_data_file("English_Ekegusii_Web_News.csv"))
    a2_bible = pd.read_csv(find_data_file("bibile.csv"))
    
    base_model2 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    a2_model = get_peft_model(base_model2, peft_config)
    
    # Stage 1: Bilingual
    ds2_1 = Dataset.from_pandas(a2_bi).map(lambda x: preprocess_nmt_function(x, "English", "Ekegusii"), batched=True)
    args2_1 = Seq2SeqTrainingArguments(output_dir="./output_arch2_s1", per_device_train_batch_size=32, num_train_epochs=1, bf16=torch.cuda.is_bf16_supported(), report_to="none")
    Seq2SeqTrainer(model=a2_model, args=args2_1, train_dataset=ds2_1, data_collator=DataCollatorForSeq2Seq(tokenizer, model=a2_model)).train()
    
    # Stage 2: Bible
    ds2_2 = Dataset.from_pandas(a2_bible.sample(min(4000, len(a2_bible)))).map(lambda x: preprocess_nmt_function(x, "English", "Ekegusii"), batched=True)
    args2_2 = Seq2SeqTrainingArguments(output_dir="./output_arch2_s2", per_device_train_batch_size=32, num_train_epochs=1, bf16=torch.cuda.is_bf16_supported(), report_to="none")
    Seq2SeqTrainer(model=a2_model, args=args2_2, train_dataset=ds2_2, data_collator=DataCollatorForSeq2Seq(tokenizer, model=a2_model)).train()
    
    # Stage 3: PSA Adaptation
    ds2_3 = Dataset.from_pandas(a1_train).map(lambda x: preprocess_nmt_function(x, "English", "Ekegusii"), batched=True)
    ds2_val = Dataset.from_pandas(a1_val).map(lambda x: preprocess_nmt_function(x, "English", "Ekegusii"), batched=True)
    args2_3 = Seq2SeqTrainingArguments(output_dir="./output_arch2_s3", eval_strategy="epoch", save_strategy="epoch", per_device_train_batch_size=32, per_device_eval_batch_size=32, num_train_epochs=3, predict_with_generate=True, bf16=torch.cuda.is_bf16_supported(), report_to="none")
    trainer2 = Seq2SeqTrainer(model=a2_model, args=args2_3, train_dataset=ds2_3, eval_dataset=ds2_val, data_collator=DataCollatorForSeq2Seq(tokenizer, model=a2_model), compute_metrics=compute_metrics)
    trainer2.train()
    
    bleu2, chrf2 = evaluate_on_test(a2_model, a1_test)
    print(f"🏆 Architecture 2 Test Results: SacreBLEU = {bleu2:.2f} | chrF = {chrf2:.2f}")
    
    # --- ARCHITECTURE 3 ---
    print("\n=== RUNNING ARCHITECTURE 3: Progressive Curriculum Transfer Learning ===")
    a3_uni = pd.read_csv(find_data_file("english_unilingual.csv"))
    
    base_model3 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    a3_model = get_peft_model(base_model3, peft_config)
    
    # Stage 1: Unilingual Adaptation
    uni_sample = a3_uni.sample(min(4000, len(a3_uni)))
    uni_sample["Ekegusii"] = uni_sample["English"]
    ds3_1 = Dataset.from_pandas(uni_sample).map(lambda x: preprocess_nmt_function(x, "English", "Ekegusii"), batched=True)
    args3_1 = Seq2SeqTrainingArguments(output_dir="./output_arch3_s1", per_device_train_batch_size=32, num_train_epochs=1, bf16=torch.cuda.is_bf16_supported(), report_to="none")
    Seq2SeqTrainer(model=a3_model, args=args3_1, train_dataset=ds3_1, data_collator=DataCollatorForSeq2Seq(tokenizer, model=a3_model)).train()
    
    # Stage 2 & 3: Bilingual & Bible
    Seq2SeqTrainer(model=a3_model, args=args2_1, train_dataset=ds2_1, data_collator=DataCollatorForSeq2Seq(tokenizer, model=a3_model)).train()
    Seq2SeqTrainer(model=a3_model, args=args2_2, train_dataset=ds2_2, data_collator=DataCollatorForSeq2Seq(tokenizer, model=a3_model)).train()
    
    # Stage 4: PSA Adaptation
    args3_4 = Seq2SeqTrainingArguments(output_dir="./output_arch3_s4", eval_strategy="epoch", save_strategy="epoch", per_device_train_batch_size=32, per_device_eval_batch_size=32, num_train_epochs=3, predict_with_generate=True, bf16=torch.cuda.is_bf16_supported(), report_to="none")
    trainer3 = Seq2SeqTrainer(model=a3_model, args=args3_4, train_dataset=ds2_3, eval_dataset=ds2_val, data_collator=DataCollatorForSeq2Seq(tokenizer, model=a3_model), compute_metrics=compute_metrics)
    trainer3.train()
    
    bleu3, chrf3 = evaluate_on_test(a3_model, a1_test)
    print(f"🏆 Architecture 3 Test Results: SacreBLEU = {bleu3:.2f} | chrF = {chrf3:.2f}")
    
    # Benchmark table output
    df_results = pd.DataFrame([
        {"Architecture": "Arch 1 (Direct PSA)", "SacreBLEU": round(bleu1, 2), "chrF": round(chrf1, 2)},
        {"Architecture": "Arch 2 (Bilingual -> Bible -> PSA)", "SacreBLEU": round(bleu2, 2), "chrF": round(chrf2, 2)},
        {"Architecture": "Arch 3 (Curriculum Transfer)", "SacreBLEU": round(bleu3, 2), "chrF": round(chrf3, 2)}
    ])
    df_results.to_csv("data_splits/benchmark_architecture_results.csv", index=False)
    print("\n=== FINAL BENCHMARK SUMMARY ===")
    print(df_results)

if __name__ == "__main__":
    train_and_evaluate()
