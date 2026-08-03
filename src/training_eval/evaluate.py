"""
evaluate.py
-----------
Evaluates Fine-Tuned NMT Models (MarianMT & Meta NLLB-200) for Swahili & Ekegusii.
Computes automatic quantitative evaluation metrics:
  - SacreBLEU (standardized BLEU)
  - chrF (character n-gram F-score)
  - COMET (neural, reference+source-aware quality estimate)
"""

import os, sys, io
from datetime import datetime, timezone
import pandas as pd
import torch
import evaluate
from transformers import MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Proper held-out test splits (produced by preprocess.py / preprocess_ekegusii.py),
# never used during training. Do NOT fall back to the full master/language datasets
# here -- those overlap with the training data and would leak into the score.
DEFAULT_TEST_DATA = {
    "Kiswahili": os.path.join("data", "intermediate", "test.csv"),
    "Ekegusii": os.path.join("data", "intermediate", "test_guz.csv"),
}

RESULTS_PATH = os.path.join("reports", "week4_automatic_metrics.csv")


def evaluate_model(model_path, test_data_path, target_lang="Kiswahili"):
    print("=" * 80)
    print(f"  EVALUATING MODEL: {model_path} (Target: {target_lang})")
    print("=" * 80)

    if not os.path.exists(model_path):
        print(f"[!] Warning: Model directory {model_path} does not exist. Run training first.")
        return None

    if not test_data_path or not os.path.exists(test_data_path):
        test_data_path = DEFAULT_TEST_DATA.get(target_lang, test_data_path)
        print(f"[!] Test data not found at given path. Using held-out split: {test_data_path}")

    if not os.path.exists(test_data_path):
        print(f"[!] Held-out test split {test_data_path} does not exist. Run preprocess.py first.")
        return None

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
    references_flat = [ref[0] for ref in references]

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
    print("\nComputing SacreBLEU, chrF, and COMET metrics...")
    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    comet = evaluate.load("comet")

    bleu_score = sacrebleu.compute(predictions=predictions, references=references)
    chrf_score = chrf.compute(predictions=predictions, references=references)
    comet_score = comet.compute(predictions=predictions, references=references_flat, sources=inputs)
    comet_mean = sum(comet_score["scores"]) / len(comet_score["scores"])

    print("\n" + "=" * 50)
    print(f"  EVALUATION SCORES ({target_lang})")
    print("=" * 50)
    print(f"  - SacreBLEU : {bleu_score['score']:.2f}")
    print(f"  - chrF Score: {chrf_score['score']:.2f}")
    print(f"  - COMET     : {comet_mean:.4f}")
    print("=" * 50)

    print("\nSample Translation Outputs:")
    for j in range(min(3, len(inputs))):
        print(f"  [{j+1}] Input (EN) : {inputs[j]}")
        print(f"      Ref ({target_lang[:2]}): {references[j][0]}")
        print(f"      Pred ({target_lang[:2]}): {predictions[j]}\n")

    result_row = {
        "Timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "Target_Lang": target_lang,
        "Model_Path": model_path,
        "Test_Data": test_data_path,
        "Test_Samples": len(inputs),
        "SacreBLEU": round(bleu_score["score"], 2),
        "chrF": round(chrf_score["score"], 2),
        "COMET": round(comet_mean, 4),
    }
    save_result(result_row)
    return result_row


def save_result(result_row):
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    row_df = pd.DataFrame([result_row])
    if os.path.exists(RESULTS_PATH):
        row_df.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(RESULTS_PATH, index=False)
    print(f"Saved result to {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned MT Models")
    parser.add_argument("--model_path", type=str, default="models/psa-en-sw-finetuned", help="Path to fine-tuned model")
    parser.add_argument("--test_data", type=str, default=None, help="Path to held-out test dataset (defaults to data/intermediate/test.csv or test_guz.csv based on --target_lang)")
    parser.add_argument("--target_lang", type=str, default="Kiswahili", help="Target language (Kiswahili or Ekegusii)")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data, args.target_lang)

if __name__ == "__main__":
    main()
