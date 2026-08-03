import pandas as pd
import os

def create_master_datasets():
    clean_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
    
    print("=== Creating Consolidated Master Datasets (30,000+ rows) ===")
    
    # 1. Master English - Ekegusii
    eng_guz_files = [
        os.path.join(clean_dir, "English_Ekegusii_Parallel_Bible.csv"),
        os.path.join(clean_dir, "PSA_Eng_Ekegusii_Clean.csv"),
        os.path.join(clean_dir, "African_Storybooks_English_Ekegusii.csv")
    ]
    dfs = []
    for f in eng_guz_files:
        if os.path.exists(f):
            d = pd.read_csv(f)
            # normalize column names
            d.columns = [c.lower().strip() for c in d.columns]
            if 'english' in d.columns and 'ekegusii' in d.columns:
                dfs.append(d[['english', 'ekegusii']])
                
    master_eng_guz = pd.concat(dfs, ignore_index=True).drop_duplicates().dropna()
    out_eng_guz = os.path.join(clean_dir, "Master_English_Ekegusii_Parallel_40k.csv")
    master_eng_guz.to_csv(out_eng_guz, index=False, encoding='utf-8-sig')
    print(f"[OK] Created Master English-Ekegusii: {len(master_eng_guz)} clean rows -> {out_eng_guz}")

    # 2. Master Ekegusii - Swahili
    guz_sw_files = [
        os.path.join(clean_dir, "Ekegusii_Swahili_Parallel_Bible.csv"),
        os.path.join(clean_dir, "African_Storybooks_Ekegusii_Swahili.csv")
    ]
    # Add Ekegusii and Kiswahili from PSA_Trilingual_Clean if available
    psa_tri = os.path.join(clean_dir, "PSA_Trilingual_Clean.csv")
    if os.path.exists(psa_tri):
        pt = pd.read_csv(psa_tri)
        pt.columns = [c.lower().strip() for c in pt.columns]
        if 'ekegusii' in pt.columns and 'kiswahili' in pt.columns:
            pt_renamed = pt[['ekegusii', 'kiswahili']].rename(columns={'kiswahili': 'swahili'})
            guz_sw_dfs = [pd.read_csv(f)[['ekegusii', 'swahili']] for f in guz_sw_files if os.path.exists(f)] + [pt_renamed]
        else:
            guz_sw_dfs = [pd.read_csv(f)[['ekegusii', 'swahili']] for f in guz_sw_files if os.path.exists(f)]
    else:
        guz_sw_dfs = [pd.read_csv(f)[['ekegusii', 'swahili']] for f in guz_sw_files if os.path.exists(f)]
        
    master_guz_sw = pd.concat(guz_sw_dfs, ignore_index=True).drop_duplicates().dropna()
    out_guz_sw = os.path.join(clean_dir, "Master_Ekegusii_Swahili_Parallel_35k.csv")
    master_guz_sw.to_csv(out_guz_sw, index=False, encoding='utf-8-sig')
    print(f"[OK] Created Master Ekegusii-Swahili: {len(master_guz_sw)} clean rows -> {out_guz_sw}")

    # 3. Master Trilingual (English - Ekegusii - Swahili)
    tri_files = [
        os.path.join(clean_dir, "Trilingual_English_Ekegusii_Swahili_Parallel_Bible.csv"),
        os.path.join(clean_dir, "African_Storybooks_Trilingual.csv")
    ]
    tri_dfs = [pd.read_csv(f)[['english', 'ekegusii', 'swahili']] for f in tri_files if os.path.exists(f)]
    
    if os.path.exists(psa_tri):
        pt = pd.read_csv(psa_tri)
        pt.columns = [c.lower().strip() for c in pt.columns]
        if 'english' in pt.columns and 'ekegusii' in pt.columns and 'kiswahili' in pt.columns:
            pt_tri = pt[['english', 'ekegusii', 'kiswahili']].rename(columns={'kiswahili': 'swahili'})
            tri_dfs.append(pt_tri)
            
    master_tri = pd.concat(tri_dfs, ignore_index=True).drop_duplicates().dropna()
    out_tri = os.path.join(clean_dir, "Master_Trilingual_Eng_Ekegusii_Swahili_34k.csv")
    master_tri.to_csv(out_tri, index=False, encoding='utf-8-sig')
    print(f"[OK] Created Master Trilingual Corpus: {len(master_tri)} clean rows -> {out_tri}")

if __name__ == "__main__":
    create_master_datasets()
