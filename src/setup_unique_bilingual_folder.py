import pandas as pd
import os
import glob

def setup_unique_bilingual_folder():
    print("=== Creating Unique Bilingual Folder (No Overlap with Trilingual Datasets & No Quran) ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    trilingual_dir = os.path.join(workspace_dir, "data_train_tringual")
    target_dir = os.path.join(workspace_dir, "data_train_bilingual")
    clean_dir = os.path.join(workspace_dir, "data", "clean")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # 0. Delete Quran dataset completely if present
    quran_file = os.path.join(clean_dir, "English_Swahili_Parallel_Quran.csv")
    if os.path.exists(quran_file):
        try:
            os.remove(quran_file)
            print(f"[REMOVED] Deleted Quran file: {quran_file}")
        except Exception as e:
            pass

    # 1. Collect all Trilingual English and Ekegusii sentence sets
    print("Loading Trilingual sentence sets to prevent cross-folder duplication...")
    trilingual_english_set = set()
    trilingual_ekegusii_set = set()
    
    for tri_name in ["bibile.csv", "psa.csv"]:
        tri_path = os.path.join(trilingual_dir, tri_name)
        if os.path.exists(tri_path):
            tdf = pd.read_csv(tri_path)
            if 'English' in tdf.columns:
                trilingual_english_set.update(tdf['English'].astype(str).str.strip().str.lower())
            if 'Ekegusii' in tdf.columns:
                trilingual_ekegusii_set.update(tdf['Ekegusii'].astype(str).str.strip().str.lower())
                
    print(f"-> Collected {len(trilingual_english_set)} trilingual English sentences to exclude.")
    
    # 2. Build unique English-Ekegusii dataset (Web News + unique pairs not in Trilingual)
    print("\n1. Processing English_Ekegusii_Web_News.csv...")
    news_path = os.path.join(clean_dir, "Web_News_RMS_English_Ekegusii.csv")
    if os.path.exists(news_path):
        ndf = pd.read_csv(news_path)
        ndf = ndf.rename(columns={'english': 'English', 'ekegusii': 'Ekegusii'})[['English', 'Ekegusii']]
        ndf = ndf.dropna().drop_duplicates().reset_index(drop=True)
        # Filter out any sentence present in trilingual
        ndf = ndf[~ndf['English'].astype(str).str.strip().str.lower().isin(trilingual_english_set)].reset_index(drop=True)
        out_news = os.path.join(target_dir, "English_Ekegusii_Web_News.csv")
        ndf.to_csv(out_news, index=False, encoding='utf-8-sig')
        print(f"[OK] Saved {len(ndf)} clean unique news rows to {out_news}")

    # 3. Build unique Swahili-Ekegusii Dictionary dataset
    print("\n2. Processing Swahili_Ekegusii_Dictionary.csv...")
    dict_paths = [
        os.path.join(clean_dir, "Kiswahili_Ekegusii_Dictionary.csv"),
        os.path.join(clean_dir, "Online_Glosbe_Swahili_Ekegusii_Dictionary.csv")
    ]
    dict_dfs = []
    for dp in dict_paths:
        if os.path.exists(dp):
            ddf = pd.read_csv(dp)
            c_s = 'swahili' if 'swahili' in ddf.columns else ('Kiswahili' if 'Kiswahili' in ddf.columns else '')
            c_g = 'ekegusii' if 'ekegusii' in ddf.columns else ('Ekegusii' if 'Ekegusii' in ddf.columns else '')
            if c_s and c_g:
                sub = ddf[[c_s, c_g]].rename(columns={c_s: 'Kiswahili', c_g: 'Ekegusii'})
                dict_dfs.append(sub)
                
    if dict_dfs:
        dict_combined = pd.concat(dict_dfs, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
        # Filter out anything present in trilingual
        dict_combined = dict_combined[~dict_combined['Ekegusii'].astype(str).str.strip().str.lower().isin(trilingual_ekegusii_set)].reset_index(drop=True)
        out_dict = os.path.join(target_dir, "Swahili_Ekegusii_Dictionary.csv")
        dict_combined.to_csv(out_dict, index=False, encoding='utf-8-sig')
        print(f"[OK] Saved {len(dict_combined)} unique dictionary entries to {out_dict}")

    # 4. Clean up any redundant bilingual files inside data_train_bilingual
    all_target_files = glob.glob(os.path.join(target_dir, "*.csv"))
    valid_names = ["English_Ekegusii_Web_News.csv", "Swahili_Ekegusii_Dictionary.csv"]
    for tf in all_target_files:
        if os.path.basename(tf) not in valid_names:
            try:
                os.remove(tf)
                print(f"[REMOVED REDUNDANT] Deleted duplicate bilingual file: {tf}")
            except Exception as e:
                pass

if __name__ == "__main__":
    setup_unique_bilingual_folder()
