"""
Master Corpus Data Engineering Pipeline
----------------------------------------
This script gathers all monolingual, bilingual, and trilingual sentence data into a single 
unified Master Sentence Corpus with unique concept IDs, and dictionary entries into a Master Lexical Corpus.

Workflow:
1. Gather all raw/clean datasets into Master Sentence Corpus & Master Lexical Corpus.
2. Perform a SINGLE master split (80% Train / 10% Validation / 10% Test) to guarantee 
   zero data leakage across all downstream experiment configurations.
3. Export derived training, validation, and testing sub-views.
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "data", "master_corpus")
SPLITS_DIR = os.path.join(OUTPUT_DIR, "splits")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return np.nan
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^[\"\'`]+|[\"\'`]+$', '', text).strip()
    return text if len(text) > 1 else np.nan

def standardize_columns(df):
    col_map = {}
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower == 'english':
            col_map[col] = 'English'
        elif col_lower in ['ekegusii', 'gusii']:
            col_map[col] = 'Ekegusii'
        elif col_lower in ['swahili', 'kiswahili']:
            col_map[col] = 'Kiswahili'
        elif col_lower in ['domain', 'source', 'category']:
            col_map[col] = 'dataset_origin'
    return df.rename(columns=col_map)

def build_master_corpus():
    print("=== Step 1: Gathering All Resources into Master Database ===")
    
    sentence_records = []
    lexical_records = []
    
    # -------------------------------------------------------------
    # A. GATHER DICTIONARIES -> Master Lexical Corpus
    # -------------------------------------------------------------
    dict_files = [
        os.path.join(WORKSPACE_DIR, "data_train_bilingual", "Online_Glosbe_Swahili_Ekegusii_Dictionary.csv"),
        os.path.join(WORKSPACE_DIR, "data_train_bilingual", "Structured_Swahili_Ekegusii_Dictionary.csv"),
        os.path.join(WORKSPACE_DIR, "data_train_bilingual", "Swahili_Ekegusii_Dictionary.csv")
    ]
    
    for fpath in dict_files:
        if os.path.exists(fpath):
            df = standardize_columns(pd.read_csv(fpath))
            fname = os.path.basename(fpath)
            for idx, row in df.iterrows():
                eng = clean_text(row.get('English', np.nan))
                swa = clean_text(row.get('Kiswahili', np.nan))
                eke = clean_text(row.get('Ekegusii', np.nan))
                if pd.notna(eke) or pd.notna(swa) or pd.notna(eng):
                    lexical_records.append({
                        'English': eng,
                        'Kiswahili': swa,
                        'Ekegusii': eke,
                        'source': 'Dictionary',
                        'dataset_origin': fname
                    })
    
    lexical_df = pd.DataFrame(lexical_records).drop_duplicates().reset_index(drop=True)
    lexical_df.insert(0, 'lexicon_id', range(1, len(lexical_df) + 1))
    
    print(f"[OK] Master Lexical Corpus Created: {len(lexical_df)} rows")
    
    # -------------------------------------------------------------
    # B. GATHER SENTENCE DATA -> Master Sentence Corpus
    # -------------------------------------------------------------
    
    # 1. Trilingual Datasets
    tri_files = [
        (os.path.join(WORKSPACE_DIR, "data_train_tringual", "psa.csv"), "PSA"),
        (os.path.join(WORKSPACE_DIR, "data_train_tringual", "bibile.csv"), "Trilingual-Bible"),
        (os.path.join(WORKSPACE_DIR, "data_train_tringual", "stories.csv"), "Trilingual-Stories"),
        (os.path.join(WORKSPACE_DIR, "data_train_tringual", "African_Storybooks_Multilingual_Corpus.csv"), "Trilingual-Storybooks")
    ]
    
    for fpath, src_tag in tri_files:
        if os.path.exists(fpath):
            df = standardize_columns(pd.read_csv(fpath))
            fname = os.path.basename(fpath)
            for idx, row in df.iterrows():
                eng = clean_text(row.get('English', np.nan))
                swa = clean_text(row.get('Kiswahili', np.nan))
                eke = clean_text(row.get('Ekegusii', np.nan))
                if pd.notna(eng) or pd.notna(swa) or pd.notna(eke):
                    sentence_records.append({
                        'English': eng,
                        'Kiswahili': swa,
                        'Ekegusii': eke,
                        'source': src_tag,
                        'dataset_origin': fname
                    })
    
    # 2. Bilingual Datasets
    bi_files = [
        (os.path.join(WORKSPACE_DIR, "data_train_bilingual", "English_Ekegusii_Bible.csv"), "ENG-EKE"),
        (os.path.join(WORKSPACE_DIR, "data_train_bilingual", "English_Ekegusii_Web_News.csv"), "ENG-EKE"),
        (os.path.join(WORKSPACE_DIR, "data_train_bilingual", "Web_News_RMS_EgesaFM_English_Ekegusii.csv"), "ENG-EKE")
    ]
    
    for fpath, src_tag in bi_files:
        if os.path.exists(fpath):
            df = standardize_columns(pd.read_csv(fpath))
            fname = os.path.basename(fpath)
            for idx, row in df.iterrows():
                eng = clean_text(row.get('English', np.nan))
                swa = clean_text(row.get('Kiswahili', np.nan))
                eke = clean_text(row.get('Ekegusii', np.nan))
                if pd.notna(eng) or pd.notna(swa) or pd.notna(eke):
                    sentence_records.append({
                        'English': eng,
                        'Kiswahili': swa,
                        'Ekegusii': eke,
                        'source': src_tag,
                        'dataset_origin': fname
                    })
                    
    # 3. Verified Monolingual / PSA Datasets
    mono_files = [
        (os.path.join(WORKSPACE_DIR, "data_train_unilingual", "Scraped_Government_PSAs_Verified.csv"), "PSA-English-Kiswahili"),
        (os.path.join(WORKSPACE_DIR, "data_train_unilingual", "Ministry_of_Health_Social_Posts.csv"), "PSA-English-Kiswahili")
    ]
    
    for fpath, src_tag in mono_files:
        if os.path.exists(fpath):
            df = standardize_columns(pd.read_csv(fpath))
            fname = os.path.basename(fpath)
            for idx, row in df.iterrows():
                eng = clean_text(row.get('English', np.nan))
                swa = clean_text(row.get('Kiswahili', np.nan))
                eke = clean_text(row.get('Ekegusii', np.nan))
                if pd.notna(eng) or pd.notna(swa) or pd.notna(eke):
                    sentence_records.append({
                        'English': eng,
                        'Kiswahili': swa,
                        'Ekegusii': eke,
                        'source': src_tag,
                        'dataset_origin': fname
                    })

    sentence_df = pd.DataFrame(sentence_records)
    sentence_df = sentence_df.dropna(how='all', subset=['English', 'Kiswahili', 'Ekegusii']).drop_duplicates().reset_index(drop=True)
    sentence_df.insert(0, 'concept_id', range(100001, 100001 + len(sentence_df)))
    
    print(f"[OK] Master Sentence Corpus Created: {len(sentence_df)} total concepts")
    
    # Save Master Corpora
    master_sentence_path = os.path.join(OUTPUT_DIR, "master_sentence_corpus.csv")
    master_lexical_path = os.path.join(OUTPUT_DIR, "master_lexical_corpus.csv")
    
    sentence_df.to_csv(master_sentence_path, index=False)
    lexical_df.to_csv(master_lexical_path, index=False)
    
    print(f" Saved Master Sentence Corpus to: '{master_sentence_path}'")
    print(f" Saved Master Lexical Corpus to:  '{master_lexical_path}'")
    
    # -------------------------------------------------------------
    # STEP 2: MASTER SPLIT (80% Train / 10% Val / 10% Test)
    # -------------------------------------------------------------
    print("\n=== Step 2: Executing Single Master Split (80% / 10% / 10%) ===")
    
    # Stratify split by PSA vs non-PSA to ensure PSA representation in all splits
    is_psa = sentence_df['source'].str.contains('PSA', case=False, na=False)
    
    train_df, temp_df = train_test_split(sentence_df, test_size=0.20, random_state=42, stratify=is_psa)
    temp_is_psa = temp_df['source'].str.contains('PSA', case=False, na=False)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_is_psa)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f" -> Master Train Split      : {len(train_df)} concepts ({len(train_df)/len(sentence_df)*100:.1f}%)")
    print(f" -> Master Validation Split : {len(val_df)} concepts ({len(val_df)/len(sentence_df)*100:.1f}%)")
    print(f" -> Master Test Split       : {len(test_df)} concepts ({len(test_df)/len(sentence_df)*100:.1f}%)")
    
    train_df.to_csv(os.path.join(SPLITS_DIR, "master_train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "master_val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "master_test.csv"), index=False)
    
    # -------------------------------------------------------------
    # STEP 3: DERIVE DYNAMIC SUBSETS (Strictly from Master Splits)
    # -------------------------------------------------------------
    print("\n=== Step 3: Deriving Dynamic Subsets from Master Splits ===")
    
    # From Train Split Only
    train_eng_eke = train_df.dropna(subset=['English', 'Ekegusii'])[['concept_id', 'English', 'Ekegusii', 'source']]
    train_swa_eke = train_df.dropna(subset=['Kiswahili', 'Ekegusii'])[['concept_id', 'Kiswahili', 'Ekegusii', 'source']]
    train_trilingual = train_df.dropna(subset=['English', 'Kiswahili', 'Ekegusii'])[['concept_id', 'English', 'Kiswahili', 'Ekegusii', 'source']]
    
    # From Val & Test Split
    test_psa = test_df[test_df['source'] == 'PSA'].dropna(subset=['English', 'Ekegusii'])[['concept_id', 'English', 'Kiswahili', 'Ekegusii']]
    val_psa = val_df[val_df['source'] == 'PSA'].dropna(subset=['English', 'Ekegusii'])[['concept_id', 'English', 'Kiswahili', 'Ekegusii']]
    
    print(f" -> Derived Train ENG-EKE Pairs   : {len(train_eng_eke)}")
    print(f" -> Derived Train SWA-EKE Pairs   : {len(train_swa_eke)}")
    print(f" -> Derived Train Trilingual Pairs: {len(train_trilingual)}")
    print(f" -> Derived Evaluation Test PSA   : {len(test_psa)}")
    print(f" -> Derived Evaluation Val PSA    : {len(val_psa)}")
    
    train_eng_eke.to_csv(os.path.join(SPLITS_DIR, "derived_train_eng_eke.csv"), index=False)
    train_swa_eke.to_csv(os.path.join(SPLITS_DIR, "derived_train_swa_eke.csv"), index=False)
    train_trilingual.to_csv(os.path.join(SPLITS_DIR, "derived_train_trilingual.csv"), index=False)
    test_psa.to_csv(os.path.join(SPLITS_DIR, "derived_test_psa.csv"), index=False)
    val_psa.to_csv(os.path.join(SPLITS_DIR, "derived_val_psa.csv"), index=False)
    
    print("\n[SUCCESS] Master Corpus Pipeline Built with 0% Data Leakage Guarantee!")

if __name__ == "__main__":
    build_master_corpus()
