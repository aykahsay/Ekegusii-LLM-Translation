"""
evaluate.py
-----------
Evaluates Fine-Tuned NMT Models (MarianMT & Meta NLLB-200) for Swahili & Ekegusii.
Computes automatic quantitative evaluation metrics:
  - BLEU & SacreBLEU
  - chrF (character n-gram F-score)
"""

import os, sys, io
import pandas as pd
import torch
import evaluate
from transformers import MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def evaluate_model(model_path, test_data_path, target_lang="Kiswahili"):
    print("=" * 80)
    print(f"  EVALUATING MODEL: {model_path} (Target: {target_lang})")
    print("=" * 80)

    if not os.path.exists(model_path):
        print(f"[!] Warning: Model directory {model_path} does not exist. Run training first.")
        return

    if not os.path.exists(test_data_path):
        test_data_path = os.path.join("data", "Master_PSA_Only.csv")
        print(f"[!] Using master dataset fallback: {test_data_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Model & Tokenizer
    if "nllb" in model_path.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="eng_Latn", tgt_lang="guz_Latn")
    else:
        tokenizer = MarianTokenizer.from_pretrained(model_path)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    # Load Test Data
    df = pd.read_csv(test_data_path, dtype=str)
    pending_marker = "N/A - Pending Fine-Tuned Model Inference"
    
    # Filter non-empty reference translations
    df_clean = df.dropna(subset=["English", target_lang])
    df_clean = df_clean[(df_clean[target_lang] != "") & (df_clean[target_lang] != pending_marker)]

    if len(df_clean) > 200:
        df_clean = df_clean.sample(200, random_state=42)

    inputs = df_clean["English"].tolist()
    references = [[ref] for ref in df_clean[target_lang].tolist()]

    print(f"Running evaluation on {len(inputs)} test samples...")
    predictions = []
    batch_size = 16

    for i in range(0, len(inputs), batch_size):
        batch_in = inputs[i:i+batch_size]
        encoded = tokenizer(batch_in, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(**encoded, max_length=128)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    # Compute Metrics
    print("\nComputing SacreBLEU and chrF metrics...")
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    bleu_score = sacrebleu.compute(predictions=predictions, references=references)
    chrf_score = chrf.compute(predictions=predictions, references=references)

    print("\n" + "=" * 50)
    print(f"  EVALUATION SCORES ({target_lang})")
    print("=" * 50)
    print(f"  • SacreBLEU : {bleu_score['score']:.2f}")
    print(f"  • chrF Score: {chrf_score['score']:.2f}")
    print("=" * 50)

    print("\nSample Translation Outputs:")
    for j in range(min(3, len(inputs))):
        print(f"  [{j+1}] Input (EN) : {inputs[j]}")
        print(f"      Ref ({target_lang[:2]}): {references[j][0]}")
        print(f"      Pred ({target_lang[:2]}): {predictions[j]}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned MT Models")
    parser.add_argument("--model_path", type=str, default="models/psa-en-sw-finetuned", help="Path to fine-tuned model")
    parser.add_argument("--test_data", type=str, default="data/languages/PSA_English_Swahili.csv", help="Path to test dataset")
    parser.add_argument("--target_lang", type=str, default="Kiswahili", help="Target language (Kiswahili or Ekegusii)")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data, args.target_lang)

if __name__ == "__main__":
    main()
