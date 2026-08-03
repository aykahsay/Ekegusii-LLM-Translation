import pandas as pd
import os

def setup_bilingual_training_folder():
    print("=== Creating data_train_bilingual Folder & Datasets (No Duplication) ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    target_dir = os.path.join(workspace_dir, "data_train_bilingual")
    clean_dir = os.path.join(workspace_dir, "data", "clean")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. English - Ekegusii (Bible + RMS Web News + PSAs + Storybooks)
    print("Building english_ekegusii.csv...")
    eng_guz_sources = [
        (os.path.join(clean_dir, "English_Ekegusii_Parallel_Bible.csv"), 'english', 'ekegusii'),
        (os.path.join(clean_dir, "Web_News_RMS_English_Ekegusii.csv"), 'english', 'ekegusii'),
        (os.path.join(clean_dir, "PSA_Eng_Ekegusii_Clean.csv"), 'English', 'Ekegusii'),
        (os.path.join(clean_dir, "African_Storybooks_English_Ekegusii.csv"), 'english', 'ekegusii')
    ]
    dfs = []
    for path, col_e, col_g in eng_guz_sources:
        if os.path.exists(path):
            d = pd.read_csv(path)
            if col_e in d.columns and col_g in d.columns:
                sub = d[[col_e, col_g]].rename(columns={col_e: 'English', col_g: 'Ekegusii'})
                dfs.append(sub)
    eng_guz_df = pd.concat(dfs, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
    out_eng_guz = os.path.join(target_dir, "english_ekegusii.csv")
    eng_guz_df.to_csv(out_eng_guz, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(eng_guz_df)} clean pairs to {out_eng_guz}")

    # 2. Ekegusii - Swahili (Bible + PSAs + Storybooks + Dictionaries)
    print("Building ekegusii_swahili.csv...")
    guz_sw_sources = [
        (os.path.join(clean_dir, "Ekegusii_Swahili_Parallel_Bible.csv"), 'ekegusii', 'swahili'),
        (os.path.join(clean_dir, "African_Storybooks_Ekegusii_Swahili.csv"), 'ekegusii', 'swahili'),
        (os.path.join(clean_dir, "Kiswahili_Ekegusii_Dictionary.csv"), 'ekegusii', 'swahili'),
        (os.path.join(clean_dir, "Online_Glosbe_Swahili_Ekegusii_Dictionary.csv"), 'ekegusii', 'swahili')
    ]
    g_dfs = []
    for path, col_g, col_s in guz_sw_sources:
        if os.path.exists(path):
            d = pd.read_csv(path)
            if col_g in d.columns and col_s in d.columns:
                sub = d[[col_g, col_s]].rename(columns={col_g: 'Ekegusii', col_s: 'Kiswahili'})
                g_dfs.append(sub)
    guz_sw_df = pd.concat(g_dfs, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
    out_guz_sw = os.path.join(target_dir, "ekegusii_swahili.csv")
    guz_sw_df.to_csv(out_guz_sw, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(guz_sw_df)} clean pairs to {out_guz_sw}")

    # 3. English - Swahili (Bible + Quran + PSAs)
    print("Building english_swahili.csv...")
    eng_sw_sources = [
        (os.path.join(clean_dir, "English_Swahili_Parallel_Bible.csv"), 'english', 'swahili'),
        (os.path.join(clean_dir, "English_Swahili_Parallel_Quran.csv"), 'english', 'swahili'),
        (os.path.join(clean_dir, "PSA_Eng_Swahili_Clean.csv"), 'English', 'Kiswahili')
    ]
    es_dfs = []
    for path, col_e, col_s in eng_sw_sources:
        if os.path.exists(path):
            d = pd.read_csv(path)
            if col_e in d.columns and col_s in d.columns:
                sub = d[[col_e, col_s]].rename(columns={col_e: 'English', col_s: 'Kiswahili'})
                es_dfs.append(sub)
    eng_sw_df = pd.concat(es_dfs, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
    out_eng_sw = os.path.join(target_dir, "english_swahili.csv")
    eng_sw_df.to_csv(out_eng_sw, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(eng_sw_df)} clean pairs to {out_eng_sw}")

if __name__ == "__main__":
    setup_bilingual_training_folder()
