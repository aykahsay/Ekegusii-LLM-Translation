import pandas as pd
import os
import glob

def setup_monolingual_psa_folder():
    print("=== Creating Monolingual PSA Folder & Datasets (No Duplication) ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    target_dir = os.path.join(workspace_dir, "data_train_monolingual")
    clean_dir = os.path.join(workspace_dir, "data", "clean")
    raw_dir = os.path.join(workspace_dir, "data", "raw")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # 1. English Monolingual PSA
    print("Building english_psa.csv...")
    eng_sources = [
        os.path.join(clean_dir, "NDMA_English_Only.csv"),
        os.path.join(clean_dir, "PSA_Trilingual_Clean.csv"),
        os.path.join(clean_dir, "PSA_Eng_Ekegusii_Clean.csv"),
        os.path.join(clean_dir, "PSA_Eng_Swahili_Clean.csv"),
        os.path.join(raw_dir, "scraped_psas_english.csv")
    ]
    eng_dfs = []
    for p in eng_sources:
        if os.path.exists(p):
            d = pd.read_csv(p)
            col_e = 'English' if 'English' in d.columns else ('english' if 'english' in d.columns else '')
            col_dom = 'Domain' if 'Domain' in d.columns else ('domain' if 'domain' in d.columns else '')
            if col_e:
                sub = pd.DataFrame()
                sub['English'] = d[col_e]
                sub['Domain'] = d[col_dom] if col_dom else 'Public Service'
                eng_dfs.append(sub)
                
    eng_df = pd.concat(eng_dfs, ignore_index=True).dropna(subset=['English']).drop_duplicates(subset=['English']).reset_index(drop=True)
    out_eng = os.path.join(target_dir, "english_psa.csv")
    eng_df.to_csv(out_eng, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(eng_df)} unique English PSA rows -> {out_eng}")

    # 2. Kiswahili Monolingual PSA
    print("Building swahili_psa.csv...")
    sw_sources = [
        os.path.join(clean_dir, "PSA_Trilingual_Clean.csv"),
        os.path.join(clean_dir, "PSA_Eng_Swahili_Clean.csv")
    ]
    sw_dfs = []
    for p in sw_sources:
        if os.path.exists(p):
            d = pd.read_csv(p)
            col_s = 'Kiswahili' if 'Kiswahili' in d.columns else ('swahili' if 'swahili' in d.columns else '')
            col_dom = 'Domain' if 'Domain' in d.columns else ('domain' if 'domain' in d.columns else '')
            if col_s:
                sub = pd.DataFrame()
                sub['Kiswahili'] = d[col_s]
                sub['Domain'] = d[col_dom] if col_dom else 'Public Service'
                sw_dfs.append(sub)
                
    sw_df = pd.concat(sw_dfs, ignore_index=True).dropna(subset=['Kiswahili']).drop_duplicates(subset=['Kiswahili']).reset_index(drop=True)
    out_sw = os.path.join(target_dir, "swahili_psa.csv")
    sw_df.to_csv(out_sw, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(sw_df)} unique Kiswahili PSA rows -> {out_sw}")

    # 3. Ekegusii Monolingual PSA
    print("Building ekegusii_psa.csv...")
    guz_sources = [
        os.path.join(clean_dir, "PSA_Trilingual_Clean.csv"),
        os.path.join(clean_dir, "PSA_Eng_Ekegusii_Clean.csv")
    ]
    guz_dfs = []
    for p in guz_sources:
        if os.path.exists(p):
            d = pd.read_csv(p)
            col_g = 'Ekegusii' if 'Ekegusii' in d.columns else ('ekegusii' if 'ekegusii' in d.columns else '')
            col_dom = 'Domain' if 'Domain' in d.columns else ('domain' if 'domain' in d.columns else '')
            if col_g:
                sub = pd.DataFrame()
                sub['Ekegusii'] = d[col_g]
                sub['Domain'] = d[col_dom] if col_dom else 'Public Service'
                guz_dfs.append(sub)
                
    guz_df = pd.concat(guz_dfs, ignore_index=True).dropna(subset=['Ekegusii']).drop_duplicates(subset=['Ekegusii']).reset_index(drop=True)
    out_guz = os.path.join(target_dir, "ekegusii_psa.csv")
    guz_df.to_csv(out_guz, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved {len(guz_df)} unique Ekegusii PSA rows -> {out_guz}")

if __name__ == "__main__":
    setup_monolingual_psa_folder()
