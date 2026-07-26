"""
build_final_datasets.py
-----------------------
Builds two clean, labelled datasets from the PSA verification results:

  data/Mixed_data.csv  -- ALL rows (PSA + Non-PSA), label column added
                          label=1 -> PSA | label=0 -> Non-PSA

  data/PSA_only.csv    -- Only confirmed PSA rows (label=1)

Requires:  data/all_psas_verified.csv  (produced by check_all_psas.py)
"""

import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
VERIFIED_CSV   = os.path.join(DATA_DIR, "all_psas_verified.csv")
MIXED_OUT      = os.path.join(DATA_DIR, "Mixed_data.csv")
PSA_ONLY_OUT   = os.path.join(DATA_DIR, "PSA_only.csv")

# Candidate text column names (tried in order)
TEXT_COLUMNS = ["English", "text", "content", "message", "body",
                "psa_text", "english_text", "english",
                "translated_text", "translation"]

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  BUILDING FINAL DATASETS")
print("=" * 65)

if not os.path.exists(VERIFIED_CSV):
    print(f"\n[ERROR] '{VERIFIED_CSV}' not found.")
    print("  Run check_all_psas.py first to generate it.")
    sys.exit(1)

df = pd.read_csv(VERIFIED_CSV, dtype=str)
print(f"\n[OK] Loaded {len(df)} rows from all_psas_verified.csv")

# Convert Is_PSA to int label
df["label"] = df["Is_PSA"].map({"True": 1, "False": 0, True: 1, False: 0})
df["label"] = df["label"].fillna(0).astype(int)

# ── Identify the main text column ─────────────────────────────────────────────
text_col = None
for col in TEXT_COLUMNS:
    if col in df.columns:
        text_col = col
        break
if text_col is None:
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    text_col = obj_cols[0] if obj_cols else None

print(f"[OK] Primary text column: '{text_col}'")

# ── Build Mixed_data ──────────────────────────────────────────────────────────
# Keep: Source_File, text column, PSA_Probability, label
keep_cols = [c for c in ["Source_File", text_col, "PSA_Probability", "label"]
             if c and c in df.columns]

mixed = df[keep_cols].copy()
mixed.rename(columns={text_col: "text"}, inplace=True)
mixed["PSA_Probability"] = mixed["PSA_Probability"].astype(float).round(4)

# Shuffle for good measure
mixed = mixed.sample(frac=1, random_state=42).reset_index(drop=True)

n_psa     = (mixed["label"] == 1).sum()
n_non_psa = (mixed["label"] == 0).sum()

mixed.to_csv(MIXED_OUT, index=False, encoding="utf-8")

print(f"\n[Mixed_data.csv]")
print(f"  Total rows : {len(mixed)}")
print(f"  PSA  (1)   : {n_psa}  ({100*n_psa/len(mixed):.1f}%)")
print(f"  Non-PSA (0): {n_non_psa}  ({100*n_non_psa/len(mixed):.1f}%)")
print(f"  Saved  -> {MIXED_OUT}")

# ── Build PSA_only ────────────────────────────────────────────────────────────
psa_only = mixed[mixed["label"] == 1].copy().reset_index(drop=True)
psa_only.to_csv(PSA_ONLY_OUT, index=False, encoding="utf-8")

print(f"\n[PSA_only.csv]")
print(f"  Total rows : {len(psa_only)}")
print(f"  Avg PSA prob: {psa_only['PSA_Probability'].mean():.4f}")
print(f"  Saved  -> {PSA_ONLY_OUT}")

# ── Quick preview ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  MIXED_DATA SAMPLE (first 5 rows)")
print("=" * 65)
sample = mixed.head(5)[["label", "PSA_Probability", "text"]]
sample["text"] = sample["text"].astype(str).str[:80]
print(sample.to_string(index=False))

print("\n" + "=" * 65)
print("  PSA_ONLY SAMPLE (first 5 rows)")
print("=" * 65)
sample2 = psa_only.head(5)[["label", "PSA_Probability", "text"]]
sample2["text"] = sample2["text"].astype(str).str[:80]
print(sample2.to_string(index=False))

print("\n[DONE] Both datasets are ready.")
