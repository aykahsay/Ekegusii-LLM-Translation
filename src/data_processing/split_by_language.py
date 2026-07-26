"""
split_by_language.py
--------------------
Enforces clean, strictly validated language-based parallel datasets without any N/A placeholders:

  1. data/languages/PSA_English_Swahili.csv
     -> Contains ONLY valid English-Swahili parallel pairs (no N/A or pending rows).
  
  2. data/languages/PSA_English_Ekegusii.csv
     -> Contains ONLY valid English-Ekegusii parallel pairs (no N/A or pending rows).

  3. data/languages/PSA_Trilingual_Complete.csv
     -> Contains ONLY complete 3-way English + Swahili + Ekegusii triplets.

  4. data/languages/PSA_Pending_Translation.csv
     -> Contains items awaiting translation.
"""

import sys, io, os
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = "data"
MASTER_PSA = os.path.join(DATA_DIR, "Master_PSA_Only.csv")
LANG_DIR = os.path.join(DATA_DIR, "languages")

PENDING = "N/A - Pending Fine-Tuned Model Inference"

def is_valid(series):
    return series.fillna('').astype(str).str.strip().ne('') & (series != PENDING) & (series.str.lower() != 'n/a')

def split():
    print("=" * 80)
    print("  BUILDING STRICTLY VALIDATED LANGUAGE PARALLEL DATASETS")
    print("=" * 80)

    if not os.path.exists(MASTER_PSA):
        print(f"[!] Error: {MASTER_PSA} not found.")
        sys.exit(1)

    os.makedirs(LANG_DIR, exist_ok=True)

    df = pd.read_csv(MASTER_PSA, dtype=str)
    print(f"Loaded {len(df):,} total records from {MASTER_PSA}")

    # Identify valid Swahili and valid Ekegusii rows
    valid_sw_mask = is_valid(df['Kiswahili'])
    valid_guz_mask = is_valid(df['Ekegusii'])

    # 1. Clean English-Swahili Parallel Dataset (Strictly NO N/A)
    en_sw_df = df[valid_sw_mask][['English', 'Kiswahili', 'Domain']].drop_duplicates(subset=['English']).reset_index(drop=True)
    en_sw_path = os.path.join(LANG_DIR, "PSA_English_Swahili.csv")
    en_sw_df.to_csv(en_sw_path, index=False, encoding="utf-8")

    # 2. Clean English-Ekegusii Parallel Dataset (Strictly NO N/A)
    en_guz_df = df[valid_guz_mask][['English', 'Ekegusii', 'Domain']].drop_duplicates(subset=['English']).reset_index(drop=True)
    en_guz_path = os.path.join(LANG_DIR, "PSA_English_Ekegusii.csv")
    en_guz_df.to_csv(en_guz_path, index=False, encoding="utf-8")

    # 3. Trilingual Complete Dataset (Strictly NO N/A in either language)
    trilingual_df = df[valid_sw_mask & valid_guz_mask][['English', 'Kiswahili', 'Ekegusii', 'Domain']].drop_duplicates(subset=['English']).reset_index(drop=True)
    trilingual_path = os.path.join(LANG_DIR, "PSA_Trilingual_Complete.csv")
    trilingual_df.to_csv(trilingual_path, index=False, encoding="utf-8")

    # 4. Pending Dataset (Items needing translation)
    pending_df = df[~valid_sw_mask | ~valid_guz_mask][['English', 'Kiswahili', 'Domain']].reset_index(drop=True)
    pending_path = os.path.join(LANG_DIR, "PSA_Pending_Translation.csv")
    pending_df.to_csv(pending_path, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("  STRICTLY VALIDATED DATASETS SAVED IN data/languages/")
    print("=" * 80)
    print(f"📁 1. PSA_English_Swahili.csv     : {len(en_sw_df):>5} parallel pairs (0 N/A) -> {en_sw_path}")
    print(f"📁 2. PSA_English_Ekegusii.csv    : {len(en_guz_df):>5} parallel pairs (0 N/A) -> {en_guz_path}")
    print(f"📁 3. PSA_Trilingual_Complete.csv : {len(trilingual_df):>5} 3-way triplets (0 N/A) -> {trilingual_path}")
    print(f"📁 4. PSA_Pending_Translation.csv : {len(pending_df):>5} pending items   -> {pending_path}")

    # Verify zero N/A in PSA_English_Swahili.csv
    na_count_sw = (en_sw_df['Kiswahili'] == PENDING).sum()
    print(f"\nVerification: N/A count in PSA_English_Swahili.csv = {na_count_sw}")
    print("[DONE]")

if __name__ == "__main__":
    split()
