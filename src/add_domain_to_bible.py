import pandas as pd
import os

def process_bible_domain_and_master():
    clean_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
    final_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\final_data"
    
    bible_path = os.path.join(clean_dir, "Trilingual_English_Ekegusii_Swahili_Parallel_Bible.csv")
    print(f"1. Loading {bible_path}...")
    bible_df = pd.read_csv(bible_path)
    
    # Standardize columns: English, Kiswahili, Ekegusii, Domain
    bible_df = bible_df.rename(columns={
        'english': 'English',
        'swahili': 'Kiswahili',
        'ekegusii': 'Ekegusii'
    })
    
    # Add Domain column
    bible_df['Domain'] = 'Religion'
    
    # Reorder columns to match Trilingual_English_Swahili_Ekegusii_Dataset.csv
    bible_df = bible_df[['English', 'Kiswahili', 'Ekegusii', 'Domain']]
    
    # Save back to clean folder
    out_bible_path = os.path.join(clean_dir, "Trilingual_English_Ekegusii_Swahili_Parallel_Bible.csv")
    bible_df.to_csv(out_bible_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Added 'Domain' = 'Religion' to Bible dataset: {len(bible_df)} rows saved to {out_bible_path}")
    
    # 2. Also save to final_data directory as Trilingual_Bible_Dataset.csv
    os.makedirs(final_dir, exist_ok=True)
    out_final_bible = os.path.join(final_dir, "Trilingual_Bible_Dataset.csv")
    bible_df.to_csv(out_final_bible, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved to final_data: {out_final_bible}")
    
    # 3. Create a Consolidated Multi-Domain Trilingual Master Dataset in final_data
    psa_path = os.path.join(final_dir, "Trilingual_English_Swahili_Ekegusii_Dataset.csv")
    storybook_path = os.path.join(clean_dir, "African_Storybooks_Trilingual.csv")
    
    all_dfs = [bible_df]
    
    if os.path.exists(psa_path):
        psa_df = pd.read_csv(psa_path)
        all_dfs.append(psa_df[['English', 'Kiswahili', 'Ekegusii', 'Domain']])
        
    if os.path.exists(storybook_path):
        sb_df = pd.read_csv(storybook_path)
        sb_df = sb_df.rename(columns={'english': 'English', 'swahili': 'Kiswahili', 'ekegusii': 'Ekegusii'})
        sb_df['Domain'] = 'Literature'
        all_dfs.append(sb_df[['English', 'Kiswahili', 'Ekegusii', 'Domain']])
        
    master_df = pd.concat(all_dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    out_master_multidomain = os.path.join(final_dir, "Master_Multidomain_Trilingual_Dataset.csv")
    master_df.to_csv(out_master_multidomain, index=False, encoding='utf-8-sig')
    print(f"[OK] Created Master Multi-Domain Dataset ({len(master_df)} rows) saved to {out_master_multidomain}")

if __name__ == "__main__":
    process_bible_domain_and_master()
