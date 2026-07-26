"""
generate_human_eval_dataset.py
-------------------------------
Generates data/human_eval_100_sentences.csv containing 100 diverse, representative
PSAs across all domains (Health, Security, Education, Governance, Agriculture)
for Native Speaker Human Evaluation of Fluency, Adequacy, and Cultural Accuracy.
"""

import sys, io, os
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

MASTER_PSA = os.path.join("data", "Master_PSA_Only.csv")
OUTPUT_PATH = os.path.join("data", "human_eval_100_sentences.csv")

def generate():
    print("=" * 80)
    print("  GENERATING HUMAN EVALUATION BENCHMARK DATASET (100 SENTENCES)")
    print("=" * 80)

    if not os.path.exists(MASTER_PSA):
        print(f"[!] File not found: {MASTER_PSA}")
        return

    df = pd.read_csv(MASTER_PSA, dtype=str)
    
    # Filter rows that have both Swahili and Ekegusii translations
    pending = "N/A - Pending Fine-Tuned Model Inference"
    valid_df = df[
        (df['Kiswahili'] != pending) & (df['Kiswahili'].fillna('') != '') &
        (df['Ekegusii'] != pending) & (df['Ekegusii'].fillna('') != '')
    ].copy()

    # Stratified sampling across domains
    sampled_rows = []
    domains = valid_df['Domain'].unique()
    samples_per_domain = max(1, 100 // len(domains))

    for dom in domains:
        dom_df = valid_df[valid_df['Domain'] == dom]
        n_sample = min(len(dom_df), samples_per_domain)
        sampled_rows.append(dom_df.sample(n_sample, random_state=42))

    eval_df = pd.concat(sampled_rows, ignore_index=True)
    
    # Fill remaining to reach exactly 100 if needed
    if len(eval_df) < 100:
        remainder = valid_df[~valid_df['English'].isin(eval_df['English'])]
        extra = remainder.sample(min(len(remainder), 100 - len(eval_df)), random_state=42)
        eval_df = pd.concat([eval_df, extra], ignore_index=True)

    eval_df = eval_df.sample(frac=1, random_state=42).reset_index(drop=True)
    eval_df['ID'] = [f"HUMAN-EVAL-{i+1:03d}" for i in range(len(eval_df))]

    # Add rating columns for Human Evaluators
    eval_df['Fluency_1to5'] = ""
    eval_df['Adequacy_1to5'] = ""
    eval_df['Cultural_Accuracy_1to5'] = ""
    eval_df['Evaluator_Notes'] = ""

    cols = ['ID', 'English', 'Kiswahili', 'Ekegusii', 'Domain', 'Fluency_1to5', 'Adequacy_1to5', 'Cultural_Accuracy_1to5', 'Evaluator_Notes']
    eval_df = eval_df[cols]

    eval_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"✔ Successfully generated human evaluation dataset: {len(eval_df)} rows")
    print(f"  Saved -> {OUTPUT_PATH}")

if __name__ == "__main__":
    generate()
