import pandas as pd
import os
import glob
import shutil

def setup_unilingual_folder():
    print("=== Creating data_train_unilingual Folder & Datasets (Strict No Duplication) ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    target_dir = os.path.join(workspace_dir, "data_train_unilingual")
    old_mono_dir = os.path.join(workspace_dir, "data_train_monolingual")
    clean_dir = os.path.join(workspace_dir, "data", "clean")
    raw_dir = os.path.join(workspace_dir, "data", "raw")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. English Unilingual (PSAs + Advisories + ReliefWeb Kenya + Scraped Text)
    print("Building English Unilingual Dataset...")
    eng_sources = [
        os.path.join(raw_dir, "reliefweb_kenya_psas.csv"),
        os.path.join(clean_dir, "NDMA_English_Only.csv"),
        os.path.join(clean_dir, "PSA_Trilingual_Clean.csv"),
        os.path.join(clean_dir, "PSA_Eng_Ekegusii_Clean.csv"),
        os.path.join(clean_dir, "PSA_Eng_Swahili_Clean.csv"),
        os.path.join(raw_dir, "scraped_psas_english.csv"),
        os.path.join(raw_dir, "NDMA_English_scraped.csv")
    ]
    eng_dfs = []
    for p in eng_sources:
        if os.path.exists(p):
            d = pd.read_csv(p)
            c_e = 'English' if 'English' in d.columns else ('english' if 'english' in d.columns else '')
            c_d = 'Domain' if 'Domain' in d.columns else ('domain' if 'domain' in d.columns else '')
            if c_e:
                sub = pd.DataFrame()
                sub['English'] = d[c_e].astype(str).str.strip()
                sub['Domain'] = d[c_d] if c_d else 'Public Service'
                eng_dfs.append(sub)
                
    eng_df = pd.concat(eng_dfs, ignore_index=True).dropna(subset=['English']).drop_duplicates(subset=['English']).reset_index(drop=True)
    # Exclude empty or bad strings
    eng_df = eng_df[eng_df['English'].str.len() > 3].reset_index(drop=True)
    
    out_eng = os.path.join(target_dir, "english_unilingual.csv")
    eng_df.to_csv(out_eng, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(eng_df)} unique English unilingual rows -> {out_eng}")

    # 2. Kiswahili Unilingual (PSAs + Advisories)
    print("Building Swahili Unilingual Dataset...")
    sw_sources = [
        os.path.join(clean_dir, "PSA_Trilingual_Clean.csv"),
        os.path.join(clean_dir, "PSA_Eng_Swahili_Clean.csv"),
        os.path.join(raw_dir, "scraped_psas_translated.csv"),
        os.path.join(raw_dir, "moh_x_posts_translated.csv")
    ]
    sw_dfs = []
    for p in sw_sources:
        if os.path.exists(p):
            d = pd.read_csv(p)
            c_s = 'Kiswahili' if 'Kiswahili' in d.columns else ('swahili' if 'swahili' in d.columns else ('Swahili' if 'Swahili' in d.columns else ''))
            c_d = 'Domain' if 'Domain' in d.columns else ('domain' if 'domain' in d.columns else '')
            if c_s:
                sub = pd.DataFrame()
                sub['Kiswahili'] = d[c_s].astype(str).str.strip()
                sub['Domain'] = d[c_d] if c_d else 'Public Service'
                sw_dfs.append(sub)
                
    sw_df = pd.concat(sw_dfs, ignore_index=True).dropna(subset=['Kiswahili']).drop_duplicates(subset=['Kiswahili']).reset_index(drop=True)
    sw_df = sw_df[sw_df['Kiswahili'].str.len() > 3].reset_index(drop=True)
    
    out_sw = os.path.join(target_dir, "swahili_unilingual.csv")
    sw_df.to_csv(out_sw, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(sw_df)} unique Kiswahili unilingual rows -> {out_sw}")

    # 3. Ekegusii Unilingual (PSAs + FineWeb Web Corpus + Advisories)
    print("Building Ekegusii Unilingual Dataset...")
    guz_sources = [
        os.path.join(clean_dir, "PSA_Trilingual_Clean.csv"),
        os.path.join(clean_dir, "PSA_Eng_Ekegusii_Clean.csv"),
        os.path.join(clean_dir, "FineWeb_Ekegusii_Web_Corpus.csv")
    ]
    guz_dfs = []
    for p in guz_sources:
        if os.path.exists(p):
            d = pd.read_csv(p)
            c_g = 'Ekegusii' if 'Ekegusii' in d.columns else ('ekegusii' if 'ekegusii' in d.columns else ('ekegusii_text' if 'ekegusii_text' in d.columns else ''))
            c_d = 'Domain' if 'Domain' in d.columns else ('domain' if 'domain' in d.columns else '')
            if c_g:
                sub = pd.DataFrame()
                sub['Ekegusii'] = d[c_g].astype(str).str.strip()
                sub['Domain'] = d[c_d] if c_d else 'Public Service'
                guz_dfs.append(sub)
                
    guz_df = pd.concat(guz_dfs, ignore_index=True).dropna(subset=['Ekegusii']).drop_duplicates(subset=['Ekegusii']).reset_index(drop=True)
    guz_df = guz_df[guz_df['Ekegusii'].str.len() > 3].reset_index(drop=True)
    
    out_guz = os.path.join(target_dir, "ekegusii_unilingual.csv")
    guz_df.to_csv(out_guz, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(guz_df)} unique Ekegusii unilingual rows -> {out_guz}")

    # 4. Remove old/redundant data_train_monolingual folder if exists to avoid clutter
    if os.path.exists(old_mono_dir):
        try:
            shutil.rmtree(old_mono_dir)
            print(f"[CLEANUP] Removed old folder: {old_mono_dir}")
        except Exception as e:
            print(f"[CLEANUP NOTE] {e}")

if __name__ == "__main__":
    setup_unilingual_folder()
