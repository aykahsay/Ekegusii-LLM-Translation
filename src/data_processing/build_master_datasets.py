"""
build_master_datasets.py
------------------------
Combines ALL raw data files across the project into two clean, standardized master datasets:

  1. data/Master_Mixed_Data.csv -- Complete dataset (PSA + Non-PSA)
  2. data/Master_PSA_Only.csv   -- Confirmed PSAs only

Standardized Schemas:
  Master_Mixed_Data.csv -> [ English, Kiswahili, Ekegusii, Domain, PSA_Probability, Is_PSA ]
  Master_PSA_Only.csv   -> [ English, Kiswahili, Ekegusii, Domain ]
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

PENDING_TXT = "N/A - Pending Fine-Tuned Model Inference"

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
    s = str(val).strip().replace('\xa0', ' ')
    if s.lower() in ["nan", "none", "n/a", "null", "n/a - pending fine-tuned model inference"]:
        return ""
    return s

def build():
    print("=" * 80)
    print("  BUILDING STANDARDIZED MASTER DATASETS: MIXED & PSA ONLY")
    print("=" * 80)

    clf, vec = load_model()

    search_dirs = [DATA_DIR, RAW_DIR, INTERMEDIATE_DIR]
    
    # Store records by normalized English text
    records = {}

    for sdir in search_dirs:
        if not os.path.exists(sdir):
            continue
        for fname in os.listdir(sdir):
            if fname.startswith("Master_"):
                continue

            fpath = os.path.join(sdir, fname)
            df = None
            if fname.endswith(".csv"):
                try:
                    df = pd.read_csv(fpath, on_bad_lines="skip", dtype=str)
                except Exception:
                    continue
            elif fname.endswith(".xlsx"):
                try:
                    df = pd.read_excel(fpath, dtype=str)
                except Exception:
                    continue
            else:
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
                eng_raw = row.get(eng_col, "")
                eng = clean_str(eng_raw)
                if len(eng.split()) < MIN_WORDS:
                    continue

                key = eng.lower().strip()
                
                swa = clean_str(row.get(swa_col, "")) if swa_col else ""
                guz = clean_str(row.get(guz_col, "")) if guz_col else ""
                dom = clean_str(row.get(dom_col, "")) if dom_col else "General"
                
                prob = None
                if prob_col and pd.notna(row.get(prob_col)):
                    try:
                        prob = float(row[prob_col])
                    except:
                        prob = None

                if key not in records:
                    records[key] = {
                        "English": eng,
                        "Kiswahili": swa,
                        "Ekegusii": guz,
                        "Domain": dom if dom != "General" else "",
                        "PSA_Probability": prob
                    }
                else:
                    # Update with non-empty translation if previously empty
                    if not records[key]["Kiswahili"] and swa:
                        records[key]["Kiswahili"] = swa
                    if not records[key]["Ekegusii"] and guz:
                        records[key]["Ekegusii"] = guz
                    if not records[key]["Domain"] and dom and dom != "General":
                        records[key]["Domain"] = dom
                    if records[key]["PSA_Probability"] is None and prob is not None:
                        records[key]["PSA_Probability"] = prob

    master_list = list(records.values())
    master_df = pd.DataFrame(master_list)
    
    # Fill empty domains with General
    master_df["Domain"] = master_df["Domain"].replace("", "General")

    # Compute missing probabilities
    missing_prob_mask = master_df["PSA_Probability"].isna()
    if missing_prob_mask.any():
        texts_to_predict = master_df.loc[missing_prob_mask, "English"].tolist()
        print(f"--> Computing PSA classification for {len(texts_to_predict)} missing items...")
        X = vec.transform(texts_to_predict)
        probs = clf.predict_proba(X)[:, 1]
        master_df.loc[missing_prob_mask, "PSA_Probability"] = np.round(probs, 4)

    master_df["PSA_Probability"] = master_df["PSA_Probability"].astype(float).round(4)
    master_df["Is_PSA"] = (master_df["PSA_Probability"] >= PSA_THRESHOLD).astype(int)

    # Explicitly standardize pending / empty translations as PENDING_TXT
    master_df["Kiswahili"] = master_df["Kiswahili"].apply(lambda x: x if x else PENDING_TXT)
    master_df["Ekegusii"]  = master_df["Ekegusii"].apply(lambda x: x if x else PENDING_TXT)

    # Shuffle for reproducibility
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 1. Save Master_Mixed_Data.csv
    mixed_path = os.path.join(DATA_DIR, "Master_Mixed_Data.csv")
    master_df.to_csv(mixed_path, index=False, encoding="utf-8")

    # 2. Save Master_PSA_Only.csv (drop probability & Is_PSA columns)
    psa_only_df = master_df[master_df["Is_PSA"] == 1].copy().reset_index(drop=True)
    psa_only_df.drop(columns=["PSA_Probability", "Is_PSA"], inplace=True)
    psa_path = os.path.join(DATA_DIR, "Master_PSA_Only.csv")
    psa_only_df.to_csv(psa_path, index=False, encoding="utf-8")

    print(f"\n✔ Master_Mixed_Data.csv : {len(master_df)} rows")
    print(f"   • Swahili complete  : {(master_df['Kiswahili'] != PENDING_TXT).sum()}")
    print(f"   • Ekegusii complete : {(master_df['Ekegusii'] != PENDING_TXT).sum()}")

    print(f"\n✔ Master_PSA_Only.csv   : {len(psa_only_df)} rows")
    print(f"   • Swahili complete  : {(psa_only_df['Kiswahili'] != PENDING_TXT).sum()}")
    print(f"   • Ekegusii complete : {(psa_only_df['Ekegusii'] != PENDING_TXT).sum()}")

if __name__ == "__main__":
    build()
