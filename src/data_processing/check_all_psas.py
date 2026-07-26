"""
check_all_psas.py
-----------------
Scans EVERY data file in the project and checks whether each entry
is a true PSA using the trained ML classifier (psa_classifier.pkl +
tfidf_vectorizer.pkl).

Outputs:
  - A summary printed to the console
  - data/all_psas_verified.csv  -- every row with its PSA verdict
  - data/confirmed_psas.csv     -- only the rows confirmed as PSAs
  - data/non_psas.csv           -- rows that are NOT PSAs
"""
import sys
import io
# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
import pickle
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_PATH = "psa_classifier.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
PSA_THRESHOLD = 0.60          # probability cutoff to call something a PSA
MIN_WORD_COUNT = 5            # skip very short / empty strings

# All CSV / JSON files to scan (relative to DATA_DIR)
FILES_TO_SCAN = [
    "treasury_psas.csv",
    "extracted_pdf_psas.csv",
    "scraped_psas_english.csv",
    "scraped_psas_verified.csv",
    "scraped_psas_translated.csv",
    "scraped_psas_part4.csv",
    "real_web_psas.csv",
    "kenya_bulk_notices_extracted.csv",
    "extracted_pdf_data.csv",
    "moh_x_posts_translated.csv",
    "psa_campaigns.json",
]

# Candidate text column names (tried in order)
TEXT_COLUMNS = ["English", "text", "content", "message", "body", "psa_text",
                "english_text", "english", "translated_text", "translation"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_classifier():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Could not find '{MODEL_PATH}' or '{VECTORIZER_PATH}'. "
            "Make sure you're running this from the project root."
        )
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vec = pickle.load(f)
    print(f"[OK] Classifier loaded: {type(clf).__name__}")
    return clf, vec


def pick_text_column(df: pd.DataFrame) -> str | None:
    """Return the first column from TEXT_COLUMNS that exists in df."""
    for col in TEXT_COLUMNS:
        if col in df.columns:
            return col
    # Fallback: first object-dtype column
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    return obj_cols[0] if obj_cols else None


def load_file(filepath: str) -> pd.DataFrame | None:
    """Load a CSV or JSON file into a DataFrame."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(filepath, on_bad_lines="skip", encoding="utf-8", dtype=str)
        elif ext == ".json":
            with open(filepath, encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                df = pd.DataFrame(raw)
            elif isinstance(raw, dict):
                # Flatten one level
                rows = []
                for key, val in raw.items():
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, dict):
                                item.setdefault("_key", key)
                                rows.append(item)
                            else:
                                rows.append({"_key": key, "content": str(item)})
                    elif isinstance(val, dict):
                        val["_key"] = key
                        rows.append(val)
                    else:
                        rows.append({"_key": key, "content": str(val)})
                df = pd.DataFrame(rows)
            else:
                print(f"  [!]  Unrecognised JSON structure — skipping.")
                return None
        else:
            print(f"  [!]  Unsupported file type '{ext}' — skipping.")
            return None
    except Exception as exc:
        print(f"  [!]  Failed to read: {exc}")
        return None
    return df


def classify_rows(df: pd.DataFrame, text_col: str,
                  clf, vec, source_name: str) -> pd.DataFrame:
    """Return a new DataFrame with psa_prob and is_psa columns."""
    rows_out = []
    texts = df[text_col].fillna("").astype(str).tolist()

    # Filter empties / too-short
    valid_mask = [len(t.split()) >= MIN_WORD_COUNT for t in texts]
    valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]

    if not valid_texts:
        print(f"  [!]  No usable text rows found.")
        return pd.DataFrame()

    # Vectorize all at once for speed
    X = vec.transform(valid_texts)
    probs = clf.predict_proba(X)[:, 1]   # P(PSA)

    valid_iter = iter(zip(valid_texts, probs))
    for i, row in df.iterrows():
        text = str(row[text_col]).strip()
        if len(text.split()) < MIN_WORD_COUNT:
            continue
        t, prob = next(valid_iter)
        row_dict = row.to_dict()
        row_dict["Source_File"] = source_name
        row_dict["PSA_Probability"] = round(float(prob), 4)
        row_dict["Is_PSA"] = bool(prob >= PSA_THRESHOLD)
        rows_out.append(row_dict)

    return pd.DataFrame(rows_out)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PSA DATA VERIFICATION - ALL FILES")
    print("=" * 65)

    clf, vec = load_classifier()

    all_results: list[pd.DataFrame] = []
    summary_rows = []

    for filename in FILES_TO_SCAN:
        filepath = os.path.join(DATA_DIR, filename)
        print(f"\n[FILE]  {filename}")

        if not os.path.exists(filepath):
            print(f"  [!]  File not found -- skipping.")
            continue

        df = load_file(filepath)
        if df is None or df.empty:
            continue

        text_col = pick_text_column(df)
        if text_col is None:
            print(f"  [!]  Could not identify a text column. Columns: {list(df.columns)}")
            continue

        print(f"  --> {len(df)} rows | text column: '{text_col}'")

        result_df = classify_rows(df, text_col, clf, vec, filename)
        if result_df.empty:
            continue

        total      = len(result_df)
        n_psa      = result_df["Is_PSA"].sum()
        n_non_psa  = total - n_psa
        avg_prob   = result_df["PSA_Probability"].mean()

        print(f"  [PSA]     {n_psa:>5}  ({100*n_psa/total:.1f}%)")
        print(f"  [Non-PSA] {n_non_psa:>5}  ({100*n_non_psa/total:.1f}%)")
        print(f"  [Avg]     PSA prob: {avg_prob:.3f}")

        summary_rows.append({
            "File": filename,
            "Total_Rows": total,
            "PSA_Count": int(n_psa),
            "Non_PSA_Count": int(n_non_psa),
            "PSA_Rate_%": round(100 * n_psa / total, 1),
            "Avg_PSA_Prob": round(avg_prob, 4),
        })

        all_results.append(result_df)

    # ── Consolidate ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CONSOLIDATING RESULTS")
    print("=" * 65)

    if not all_results:
        print("No data was processed.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    confirmed = combined[combined["Is_PSA"] == True].copy()
    non_psa   = combined[combined["Is_PSA"] == False].copy()

    out_all       = os.path.join(DATA_DIR, "all_psas_verified.csv")
    out_confirmed = os.path.join(DATA_DIR, "confirmed_psas.csv")
    out_non       = os.path.join(DATA_DIR, "non_psas.csv")

    combined.to_csv(out_all, index=False, encoding="utf-8")
    confirmed.to_csv(out_confirmed, index=False, encoding="utf-8")
    non_psa.to_csv(out_non, index=False, encoding="utf-8")

    print(f"\n  Total rows processed : {len(combined)}")
    print(f"  Confirmed PSAs       : {len(confirmed)}")
    print(f"  Non-PSAs             : {len(non_psa)}")
    print(f"\n  Saved:")
    print(f"    -> {out_all}")
    print(f"    -> {out_confirmed}")
    print(f"    -> {out_non}")

    # ── Per-file summary table ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PER-FILE SUMMARY")
    print("=" * 65)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # ── Top 10 most confident PSAs overall ────────────────────────────────
    print("\n" + "=" * 65)
    print("  TOP 10 HIGHEST CONFIDENCE PSAs (across all files)")
    print("=" * 65)
    top10 = confirmed.nlargest(10, "PSA_Probability")
    for _, r in top10.iterrows():
        # Find the text column value
        text_val = ""
        for col in TEXT_COLUMNS:
            if col in r and pd.notna(r[col]):
                text_val = str(r[col])[:120]
                break
        print(f"\n  [{r['Source_File']}]  prob={r['PSA_Probability']:.4f}")
        print(f"  {text_val}{'...' if len(text_val)==120 else ''}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
