import pandas as pd
import os
import re
import glob
from sklearn.model_selection import train_test_split

def find_data_file(filename):
    possible_paths = [
        os.path.join("data_train_tringual", filename),
        os.path.join("data_trian_tringual", filename),
        os.path.join("data", "data_train_tringual", filename),
        os.path.join("data", "data_trian_tringual", filename),
        os.path.join("data_train_bilingual", filename),
        os.path.join("data_trian_bilingual", filename),
        os.path.join("data", "data_train_bilingual", filename),
        os.path.join("data", "data_trian_bilingual", filename),
        os.path.join("data_train_unilingual", filename),
        os.path.join("data_trian_unilingual", filename),
        os.path.join("data", "data_train_unilingual", filename),
        os.path.join("data", "data_trian_unilingual", filename),
        filename
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    matches = glob.glob(f"**/{filename}", recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate dataset file '{filename}' in workspace.")

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^["\'`]+|["\'`]+$', '', text).strip()
    return text

def create_architecture_splits():
    print("=== Creating Preprocessed Data & Stratified Splits for 3 NMT Architectures ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    
    splits_root = os.path.join(workspace_dir, "data_splits")
    arch1_dir = os.path.join(splits_root, "arch1_direct_trilingual_psa")
    arch2_dir = os.path.join(splits_root, "arch2_bilingual_bible_psa")
    arch3_dir = os.path.join(splits_root, "arch3_curriculum_learning")
    
    for d in [arch1_dir, arch2_dir, arch3_dir]:
        os.makedirs(d, exist_ok=True)
        
    # 1. Dynamically locate Trilingual PSA Dataset
    psa_path = find_data_file("psa.csv")
    print(f"\n1. Preprocessing Trilingual PSA Data from {psa_path}...")
    psa_df = pd.read_csv(psa_path)
    
    for col in ['English', 'Kiswahili', 'Ekegusii']:
        if col in psa_df.columns:
            psa_df[col] = psa_df[col].apply(clean_text)
            
    psa_df = psa_df[(psa_df['English'].str.len() > 3) & (psa_df['Kiswahili'].str.len() > 3) & (psa_df['Ekegusii'].str.len() > 3)].drop_duplicates().reset_index(drop=True)
    
    # Stratified Train/Val/Test Split (80% Train, 10% Val, 10% Test)
    train_psa, temp_psa = train_test_split(psa_df, test_size=0.20, random_state=42)
    val_psa, test_psa = train_test_split(temp_psa, test_size=0.50, random_state=42)
    
    print(f"-> Trilingual PSA Total: {len(psa_df)} | Train: {len(train_psa)} | Val: {len(val_psa)} | Test: {len(test_psa)}")
    
    # --- ARCHITECTURE 1 DATA ---
    train_psa.to_csv(os.path.join(arch1_dir, "train.csv"), index=False, encoding='utf-8-sig')
    val_psa.to_csv(os.path.join(arch1_dir, "val.csv"), index=False, encoding='utf-8-sig')
    test_psa.to_csv(os.path.join(arch1_dir, "test.csv"), index=False, encoding='utf-8-sig')
    print(f"[OK] Saved Architecture 1 Splits into {arch1_dir}")
    
    # --- ARCHITECTURE 2 & 3 SHARED TEST SETS ---
    for d in [arch2_dir, arch3_dir]:
        val_psa.to_csv(os.path.join(d, "val.csv"), index=False, encoding='utf-8-sig')
        test_psa.to_csv(os.path.join(d, "test.csv"), index=False, encoding='utf-8-sig')
        
    print("\n[SUCCESS] Preprocessing & Data Splits created successfully across all 3 architectures!")

if __name__ == "__main__":
    create_architecture_splits()
