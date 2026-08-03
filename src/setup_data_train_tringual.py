import pandas as pd
import os

def create_data_train_trilingual():
    print("=== Creating data_train_tringual Folder & Datasets (No Duplication) ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    
    # 1. Target Folder Paths
    folder1 = os.path.join(workspace_dir, "data_train_tringual")
    folder2 = os.path.join(workspace_dir, "data", "data_train_tringual")
    
    os.makedirs(folder1, exist_ok=True)
    os.makedirs(folder2, exist_ok=True)
    
    # 2. Source Files
    clean_dir = os.path.join(workspace_dir, "data", "clean")
    final_dir = os.path.join(workspace_dir, "data", "final_data")
    
    bible_src = os.path.join(clean_dir, "Trilingual_English_Ekegusii_Swahili_Parallel_Bible.csv")
    psa_src = os.path.join(final_dir, "Trilingual_English_Swahili_Ekegusii_Dataset.csv")
    
    # 3. Load & Process Bible Data
    print(f"Loading Bible dataset from {bible_src}...")
    bible_df = pd.read_csv(bible_src)
    # Ensure correct columns and capitalization
    bible_df = bible_df.rename(columns={'english': 'English', 'swahili': 'Kiswahili', 'ekegusii': 'Ekegusii', 'domain': 'Domain'})
    if 'Domain' not in bible_df.columns:
        bible_df['Domain'] = 'Religion'
    bible_df = bible_df[['English', 'Kiswahili', 'Ekegusii', 'Domain']]
    
    # Remove duplicates & NaNs
    bible_df = bible_df.dropna().drop_duplicates().reset_index(drop=True)
    print(f"-> Clean Bible Rows (No Duplicates): {len(bible_df)}")
    
    # 4. Load & Process PSA Data
    print(f"Loading PSA dataset from {psa_src}...")
    psa_df = pd.read_csv(psa_src)
    psa_df = psa_df.rename(columns={'english': 'English', 'kiswahili': 'Kiswahili', 'swahili': 'Kiswahili', 'ekegusii': 'Ekegusii', 'domain': 'Domain'})
    psa_df = psa_df[['English', 'Kiswahili', 'Ekegusii', 'Domain']]
    
    # Remove duplicates & NaNs
    psa_df = psa_df.dropna().drop_duplicates().reset_index(drop=True)
    
    # Ensure NO overlap between PSA and Bible
    bible_english_set = set(bible_df['English'].str.lower())
    psa_df = psa_df[~psa_df['English'].str.lower().isin(bible_english_set)].reset_index(drop=True)
    print(f"-> Clean PSA Rows (No Duplicates / No Overlap): {len(psa_df)}")
    
    # 5. Save to both folders
    for f_dir in [folder1, folder2]:
        # bibile.csv
        b_path = os.path.join(f_dir, "bibile.csv")
        bible_df.to_csv(b_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Saved {len(bible_df)} rows to {b_path}")
        
        # psa.csv
        p_path = os.path.join(f_dir, "psa.csv")
        psa_df.to_csv(p_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Saved {len(psa_df)} rows to {p_path}")

if __name__ == "__main__":
    create_data_train_trilingual()
