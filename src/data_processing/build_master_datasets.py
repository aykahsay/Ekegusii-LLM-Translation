"""
build_master_datasets.py
------------------------
Combines ALL raw data files across the project into two clean, standardized master datasets:

  1. data/Master_Mixed_Data.csv -- Complete dataset (PSA + Non-PSA)
  2. data/Master_PSA_Only.csv   -- Confirmed PSAs only (English, Kiswahili, Ekegusii, Domain, Source)

Standardized Schemas:
  Master_Mixed_Data.csv -> [ English, Kiswahili, Ekegusii, Domain, Source, PSA_Probability, Is_PSA ]
  Master_PSA_Only.csv   -> [ English, Kiswahili, Ekegusii, Domain, Source ]
"""

import sys, io, os, pickle
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")
MODEL_PATH = "psa_classifier.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
PSA_THRESHOLD = 0.60
MIN_WORDS = 4

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError("Model or Vectorizer missing.")
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vec = pickle.load(f)
    return clf, vec

def clean_str(val):
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in ["nan", "none", "n/a", "null"] else s

def build():
    print("=" * 80)
    print("  BUILDING STANDARDIZED MASTER DATASETS: MIXED & PSA ONLY")
    print("=" * 80)

    clf, vec = load_model()

    search_dirs = [DATA_DIR, RAW_DIR, INTERMEDIATE_DIR]
    all_rows = []

    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for fname in os.listdir(sdir):
            if not fname.endswith(".csv") or fname.startswith("Master_"):
                continue

            fpath = os.path.join(sdir, fname)
            try:
                df = pd.read_csv(fpath, on_bad_lines="skip", dtype=str)
            except Exception:
                continue

            eng_col = "English" if "English" in df.columns else None
            swa_col = "Kiswahili" if "Kiswahili" in df.columns else ("Swahili" if "Swahili" in df.columns else None)
            guz_col = "Ekegusii" if "Ekegusii" in df.columns else None
            dom_col = "Domain" if "Domain" in df.columns else None
            prob_col = "PSA_Probability" if "PSA_Probability" in df.columns else None

            if not eng_col:
                obj_cols = df.select_dtypes(include="object").columns.tolist()
                if obj_cols:
                    eng_col = obj_cols[0]
                else:
                    continue

            for _, row in df.iterrows():
                eng = clean_str(row.get(eng_col, ""))
                if len(eng.split()) < MIN_WORDS:
                    continue

                swa = clean_str(row.get(swa_col, "")) if swa_col else ""
                guz = clean_str(row.get(guz_col, "")) if guz_col else ""
                dom = clean_str(row.get(dom_col, "")) if dom_col else "General"
                
                source_val = fname
                if "Source" in df.columns and clean_str(row["Source"]):
                    source_val = clean_str(row["Source"])
                elif "Filename" in df.columns and clean_str(row["Filename"]):
                    source_val = clean_str(row["Filename"])

                prob = None
                if prob_col and pd.notna(row.get(prob_col)):
                    try:
                        prob = float(row[prob_col])
                    except:
                        prob = None

                all_rows.append({
                    "English": eng,
                    "Kiswahili": swa if swa else "N/A",
                    "Ekegusii": guz if guz else "N/A",
                    "Domain": dom if dom else "General",
                    "Source": source_val,
                    "PSA_Probability": prob
                })

    master_df = pd.DataFrame(all_rows)

    missing_prob_mask = master_df["PSA_Probability"].isna()
    if missing_prob_mask.any():
        texts_to_predict = master_df.loc[missing_prob_mask, "English"].tolist()
        X = vec.transform(texts_to_predict)
        probs = clf.predict_proba(X)[:, 1]
        master_df.loc[missing_prob_mask, "PSA_Probability"] = np.round(probs, 4)

    master_df["PSA_Probability"] = master_df["PSA_Probability"].astype(float).round(4)
    master_df["Is_PSA"] = (master_df["PSA_Probability"] >= PSA_THRESHOLD).astype(int)

    master_df["has_swa"] = (master_df["Kiswahili"] != "N/A").astype(int)
    master_df["has_guz"] = (master_df["Ekegusii"] != "N/A").astype(int)
    
    master_df = master_df.sort_values(
        by=["has_swa", "has_guz", "PSA_Probability"], ascending=False
    ).drop_duplicates(subset=["English"], keep="first")

    master_df.drop(columns=["has_swa", "has_guz"], inplace=True)
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 1. Save Master_Mixed_Data.csv
    mixed_path = os.path.join(DATA_DIR, "Master_Mixed_Data.csv")
    master_df.to_csv(mixed_path, index=False, encoding="utf-8")

    # 2. Save Master_PSA_Only.csv (drop probability & Is_PSA columns)
    psa_only_df = master_df[master_df["Is_PSA"] == 1].copy().reset_index(drop=True)
    psa_only_df.drop(columns=["PSA_Probability", "Is_PSA"], inplace=True)
    psa_path = os.path.join(DATA_DIR, "Master_PSA_Only.csv")
    psa_only_df.to_csv(psa_path, index=False, encoding="utf-8")

    print(f"✔ Updated Master Datasets: Mixed={len(master_df)} rows | PSA_Only={len(psa_only_df)} rows")

if __name__ == "__main__":
    build()
