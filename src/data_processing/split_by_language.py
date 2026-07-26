"""
distribute_language_datasets.py
--------------------------------
Distributes all valid records from Master_PSA_Only.csv into exact language datasets,
and completely removes PSA_Pending_Translation.csv (no N/A pending files left).

Target Output Files:
  1. data/languages/PSA_English_Swahili.csv     (All valid EN-SW pairs)
  2. data/languages/PSA_English_Ekegusii.csv    (All valid EN-GUZ pairs)
  3. data/languages/PSA_Trilingual_Complete.csv (All complete 3-way triplets)
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

PENDING_TXT = "N/A - Pending Fine-Tuned Model Inference"

def is_valid(series):
    s = series.fillna('').astype(str).str.strip()
    return (s != '') & (s != PENDING_TXT) & (s.str.lower() != 'n/a')

def run():
    print("=" * 80)
    print("  DISTRIBUTING MASTER DATASET INTO CLEAN LANGUAGE FILES (NO PENDING/NA FILES)")
    print("=" * 80)

    if not os.path.exists(MASTER_PSA):
        print(f"[!] Error: {MASTER_PSA} missing.")
        sys.exit(1)

    os.makedirs(LANG_DIR, exist_ok=True)
    df = pd.read_csv(MASTER_PSA, dtype=str)

    # Masks for clean valid translation strings
    valid_sw = is_valid(df['Kiswahili'])
    valid_guz = is_valid(df['Ekegusii'])

    # 1. English - Swahili Parallel Dataset
    en_sw_df = df[valid_sw][['English', 'Kiswahili', 'Domain']].drop_duplicates(subset=['English']).reset_index(drop=True)
    en_sw_path = os.path.join(LANG_DIR, "PSA_English_Swahili.csv")
    en_sw_df.to_csv(en_sw_path, index=False, encoding="utf-8")

    # 2. English - Ekegusii Parallel Dataset
    en_guz_df = df[valid_guz][['English', 'Ekegusii', 'Domain']].drop_duplicates(subset=['English']).reset_index(drop=True)
    en_guz_path = os.path.join(LANG_DIR, "PSA_English_Ekegusii.csv")
    en_guz_df.to_csv(en_guz_path, index=False, encoding="utf-8")

    # 3. Trilingual Complete Dataset (English + Swahili + Ekegusii)
    trilingual_df = df[valid_sw & valid_guz][['English', 'Kiswahili', 'Ekegusii', 'Domain']].drop_duplicates(subset=['English']).reset_index(drop=True)
    trilingual_path = os.path.join(LANG_DIR, "PSA_Trilingual_Complete.csv")
    trilingual_df.to_csv(trilingual_path, index=False, encoding="utf-8")

    # 4. Delete PSA_Pending_Translation.csv if it exists
    pending_path = os.path.join(LANG_DIR, "PSA_Pending_Translation.csv")
    if os.path.exists(pending_path):
        os.remove(pending_path)
        print(f"🗑️ Removed pending file: {pending_path}")

    print("\n" + "=" * 80)
    print("  FINAL LANGUAGE DATASETS SUMMARY (data/languages/)")
    print("=" * 80)
    print(f"📁 1. PSA_English_Swahili.csv     : {len(en_sw_df):>5} clean pairs   -> {en_sw_path}")
    print(f"📁 2. PSA_English_Ekegusii.csv    : {len(en_guz_df):>5} clean pairs   -> {en_guz_path}")
    print(f"📁 3. PSA_Trilingual_Complete.csv : {len(trilingual_df):>5} 3-way triplets -> {trilingual_path}")
    print("=" * 80)
    print("[DONE] No N/A pending files remaining.")

if __name__ == "__main__":
    run()
