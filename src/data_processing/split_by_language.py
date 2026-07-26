"""
split_by_language.py
--------------------
Splits data/Master_PSA_Only.csv into language-specific parallel datasets:

  data/languages/
  ├── PSA_English_Swahili.csv       (5,752 parallel EN-SW pairs)
  ├── PSA_English_Ekegusii.csv      (4,557 parallel EN-GUZ pairs)
  ├── PSA_Trilingual_Complete.csv   (2,806 complete EN-SW-GUZ triplets)
  └── PSA_Pending_Translation.csv   (4,872 items with pending translations)
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

def split():
    print("=" * 80)
    print("  SPLITTING MASTER PSA DATASET BY LANGUAGE")
    print("=" * 80)

    if not os.path.exists(MASTER_PSA):
        print(f"[!] Error: {MASTER_PSA} not found.")
        sys.exit(1)

    os.makedirs(LANG_DIR, exist_ok=True)

    df = pd.read_csv(MASTER_PSA, dtype=str)
    print(f"Loaded {len(df)} total rows from {MASTER_PSA}")

    # Masks for valid (non-pending, non-empty) translations
    has_swa = df['Kiswahili'].fillna('').astype(str).str.strip().ne('') & (df['Kiswahili'] != PENDING)
    has_guz = df['Ekegusii'].fillna('').astype(str).str.strip().ne('') & (df['Ekegusii'] != PENDING)

    # 1. English - Swahili Parallel Dataset
    en_sw_df = df[has_swa][['English', 'Kiswahili', 'Domain']].reset_index(drop=True)
    en_sw_path = os.path.join(LANG_DIR, "PSA_English_Swahili.csv")
    en_sw_df.to_csv(en_sw_path, index=False, encoding="utf-8")

    # 2. English - Ekegusii Parallel Dataset
    en_guz_df = df[has_guz][['English', 'Ekegusii', 'Domain']].reset_index(drop=True)
    en_guz_path = os.path.join(LANG_DIR, "PSA_English_Ekegusii.csv")
    en_guz_df.to_csv(en_guz_path, index=False, encoding="utf-8")

    # 3. Trilingual Complete (English + Swahili + Ekegusii)
    trilingual_df = df[has_swa & has_guz][['English', 'Kiswahili', 'Ekegusii', 'Domain']].reset_index(drop=True)
    trilingual_path = os.path.join(LANG_DIR, "PSA_Trilingual_Complete.csv")
    trilingual_df.to_csv(trilingual_path, index=False, encoding="utf-8")

    # 4. Pending Translation Dataset
    pending_df = df[~has_swa | ~has_guz][['English', 'Kiswahili', 'Ekegusii', 'Domain']].reset_index(drop=True)
    pending_path = os.path.join(LANG_DIR, "PSA_Pending_Translation.csv")
    pending_df.to_csv(pending_path, index=False, encoding="utf-8")

    print("\n" + "=" * 80)
    print("  LANGUAGE-BASED DATASETS SAVED IN data/languages/")
    print("=" * 80)
    print(f"📁 1. PSA_English_Swahili.csv     : {len(en_sw_df):>5} parallel pairs  -> {en_sw_path}")
    print(f"📁 2. PSA_English_Ekegusii.csv    : {len(en_guz_df):>5} parallel pairs  -> {en_guz_path}")
    print(f"📁 3. PSA_Trilingual_Complete.csv : {len(trilingual_df):>5} 3-way triplets -> {trilingual_path}")
    print(f"📁 4. PSA_Pending_Translation.csv : {len(pending_df):>5} pending items   -> {pending_path}")
    print("\n[DONE]")

if __name__ == "__main__":
    split()
