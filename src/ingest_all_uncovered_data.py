import pandas as pd
import os

def ingest_all_uncovered_workspace_data():
    print("=== Ingesting ALL Uncovered Data Records into 3 Active Training Folders ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    trilingual_dir = os.path.join(workspace_dir, "data_train_tringual")
    bilingual_dir = os.path.join(workspace_dir, "data_train_bilingual")
    unilingual_dir = os.path.join(workspace_dir, "data_train_unilingual")
    clean_dir = os.path.join(workspace_dir, "data", "clean")
    raw_dir = os.path.join(workspace_dir, "data", "raw")
    inter_dir = os.path.join(workspace_dir, "data", "intermediate")
    
    # 1. TRILINGUAL: Add African Storybooks (135 rows) -> stories.csv in data_train_tringual
    print("\n1. Processing Trilingual African Storybooks...")
    sb_path = os.path.join(clean_dir, "African_Storybooks_Trilingual.csv")
    if os.path.exists(sb_path):
        sb_df = pd.read_csv(sb_path)
        sb_df = sb_df.rename(columns={'english': 'English', 'swahili': 'Kiswahili', 'ekegusii': 'Ekegusii'})
        sb_df['Domain'] = 'Literature'
        sb_df = sb_df[['English', 'Kiswahili', 'Ekegusii', 'Domain']].dropna().drop_duplicates().reset_index(drop=True)
        out_stories = os.path.join(trilingual_dir, "stories.csv")
        sb_df.to_csv(out_stories, index=False, encoding='utf-8-sig')
        print(f"[OK] Created {out_stories} with {len(sb_df)} clean literature rows.")

    # 2. BILINGUAL: Add extra 4,422 English-Ekegusii Bible verses to data_train_bilingual/English_Ekegusii_Bible.csv
    print("\n2. Processing English-Ekegusii Bible Extra Verses...")
    ee_bible_path = os.path.join(clean_dir, "English_Ekegusii_Parallel_Bible.csv")
    if os.path.exists(ee_bible_path):
        ee_df = pd.read_csv(ee_bible_path)
        ee_df = ee_df.rename(columns={'english': 'English', 'ekegusii': 'Ekegusii'})[['English', 'Ekegusii']].dropna().drop_duplicates().reset_index(drop=True)
        
        # Load existing trilingual English set to keep ONLY unique pairs
        tri_eng_set = set()
        for f in [os.path.join(trilingual_dir, "bibile.csv"), os.path.join(trilingual_dir, "psa.csv")]:
            if os.path.exists(f):
                tdf = pd.read_csv(f)
                if 'English' in tdf.columns:
                    tri_eng_set.update(tdf['English'].astype(str).str.strip().str.lower())
                    
        # Filter out verses already in trilingual bibile.csv
        ee_unique = ee_df[~ee_df['English'].astype(str).str.strip().str.lower().isin(tri_eng_set)].reset_index(drop=True)
        out_ee_bible = os.path.join(bilingual_dir, "English_Ekegusii_Bible.csv")
        ee_unique.to_csv(out_ee_bible, index=False, encoding='utf-8-sig')
        print(f"[OK] Created {out_ee_bible} with {len(ee_unique)} unique English-Ekegusii Bible pairs.")

    # 3. UNILINGUAL: Append all remaining raw/intermediate PDF PSAs to english_unilingual.csv
    print("\n3. Processing Uncovered Raw & Intermediate PDF PSAs...")
    raw_files = [
        os.path.join(raw_dir, "extracted_pdf_psas.csv"),
        os.path.join(raw_dir, "high_quality_scraped_psas.csv"),
        os.path.join(raw_dir, "kenya_bulk_notices_extracted.csv"),
        os.path.join(raw_dir, "treasury_psas.csv"),
        os.path.join(raw_dir, "PSA.csv"),
        os.path.join(raw_dir, "scraped_psas_verified.csv"),
        os.path.join(inter_dir, "all_psas_verified.csv")
    ]
    
    eng_unilingual_path = os.path.join(unilingual_dir, "english_unilingual.csv")
    existing_eng_df = pd.read_csv(eng_unilingual_path) if os.path.exists(eng_unilingual_path) else pd.DataFrame()
    
    extra_dfs = [existing_eng_df]
    for rf in raw_files:
        if os.path.exists(rf):
            rdf = pd.read_csv(rf)
            col_e = 'English' if 'English' in rdf.columns else ('text' if 'text' in rdf.columns else '')
            col_d = 'Domain' if 'Domain' in rdf.columns else ''
            if col_e:
                sub = pd.DataFrame()
                sub['English'] = rdf[col_e].astype(str).str.strip()
                sub['Domain'] = rdf[col_d] if col_d else 'Public Service'
                extra_dfs.append(sub)
                
    final_eng_unilingual = pd.concat(extra_dfs, ignore_index=True).dropna(subset=['English']).drop_duplicates(subset=['English']).reset_index(drop=True)
    final_eng_unilingual = final_eng_unilingual[final_eng_unilingual['English'].str.len() > 3].reset_index(drop=True)
    
    final_eng_unilingual.to_csv(eng_unilingual_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Updated {eng_unilingual_path}: Total Unique English Rows = {len(final_eng_unilingual)}")

if __name__ == "__main__":
    ingest_all_uncovered_workspace_data()
