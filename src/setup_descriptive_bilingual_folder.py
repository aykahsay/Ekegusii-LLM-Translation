import pandas as pd
import os

def setup_descriptive_bilingual_folder():
    print("=== Creating Descriptively Named Bilingual Folder & Datasets (Excluding Quran, No Duplication) ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    target_dir = os.path.join(workspace_dir, "data_train_bilingual")
    clean_dir = os.path.join(workspace_dir, "data", "clean")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # 0. Delete Quran file if present (User requested: Quran is NOT needed at all)
    quran_file = os.path.join(clean_dir, "English_Swahili_Parallel_Quran.csv")
    if os.path.exists(quran_file):
        try:
            os.remove(quran_file)
            print(f"[REMOVED] Excluded and deleted Quran dataset: {quran_file}")
        except Exception as e:
            print(f"[NOTE] Could not remove Quran file: {e}")

    # Map of Descriptive Filename -> List of Source Files with Column mappings
    bilingual_datasets = {
        # 1. Master Parallel Corpora by Language Pair
        "English_Ekegusii_Master_Parallel.csv": [
            (os.path.join(clean_dir, "English_Ekegusii_Parallel_Bible.csv"), 'english', 'ekegusii'),
            (os.path.join(clean_dir, "Web_News_RMS_English_Ekegusii.csv"), 'english', 'ekegusii'),
            (os.path.join(clean_dir, "PSA_Eng_Ekegusii_Clean.csv"), 'English', 'Ekegusii'),
            (os.path.join(clean_dir, "African_Storybooks_English_Ekegusii.csv"), 'english', 'ekegusii')
        ],
        "Ekegusii_Swahili_Master_Parallel.csv": [
            (os.path.join(clean_dir, "Ekegusii_Swahili_Parallel_Bible.csv"), 'ekegusii', 'swahili'),
            (os.path.join(clean_dir, "African_Storybooks_Ekegusii_Swahili.csv"), 'ekegusii', 'swahili'),
            (os.path.join(clean_dir, "Kiswahili_Ekegusii_Dictionary.csv"), 'ekegusii', 'swahili'),
            (os.path.join(clean_dir, "Online_Glosbe_Swahili_Ekegusii_Dictionary.csv"), 'ekegusii', 'swahili')
        ],
        "English_Swahili_Master_Parallel.csv": [
            (os.path.join(clean_dir, "English_Swahili_Parallel_Bible.csv"), 'english', 'swahili'),
            (os.path.join(clean_dir, "PSA_Eng_Swahili_Clean.csv"), 'English', 'Kiswahili')
        ],
        
        # 2. Specific Domain Parallel Datasets
        "English_Ekegusii_Bible.csv": [
            (os.path.join(clean_dir, "English_Ekegusii_Parallel_Bible.csv"), 'english', 'ekegusii')
        ],
        "Ekegusii_Swahili_Bible.csv": [
            (os.path.join(clean_dir, "Ekegusii_Swahili_Parallel_Bible.csv"), 'ekegusii', 'swahili')
        ],
        "English_Swahili_Bible.csv": [
            (os.path.join(clean_dir, "English_Swahili_Parallel_Bible.csv"), 'english', 'swahili')
        ],
        "English_Ekegusii_Web_News.csv": [
            (os.path.join(clean_dir, "Web_News_RMS_English_Ekegusii.csv"), 'english', 'ekegusii')
        ],
        "English_Ekegusii_Public_Advisories.csv": [
            (os.path.join(clean_dir, "PSA_Eng_Ekegusii_Clean.csv"), 'English', 'Ekegusii')
        ],
        "English_Swahili_Public_Advisories.csv": [
            (os.path.join(clean_dir, "PSA_Eng_Swahili_Clean.csv"), 'English', 'Kiswahili')
        ],
        "Swahili_Ekegusii_Dictionary.csv": [
            (os.path.join(clean_dir, "Kiswahili_Ekegusii_Dictionary.csv"), 'swahili', 'ekegusii'),
            (os.path.join(clean_dir, "Online_Glosbe_Swahili_Ekegusii_Dictionary.csv"), 'swahili', 'ekegusii')
        ],
        "English_Ekegusii_Storybooks.csv": [
            (os.path.join(clean_dir, "African_Storybooks_English_Ekegusii.csv"), 'english', 'ekegusii')
        ],
        "Ekegusii_Swahili_Storybooks.csv": [
            (os.path.join(clean_dir, "African_Storybooks_Ekegusii_Swahili.csv"), 'ekegusii', 'swahili')
        ]
    }
    
    for filename, sources in bilingual_datasets.items():
        dfs = []
        for path, col1, col2 in sources:
            if os.path.exists(path):
                d = pd.read_csv(path)
                if col1 in d.columns and col2 in d.columns:
                    name1 = 'English' if 'eng' in col1.lower() else ('Ekegusii' if 'guz' in col1.lower() or 'ekegusii' in col1.lower() else 'Kiswahili')
                    name2 = 'English' if 'eng' in col2.lower() else ('Ekegusii' if 'guz' in col2.lower() or 'ekegusii' in col2.lower() else 'Kiswahili')
                    sub = d[[col1, col2]].rename(columns={col1: name1, col2: name2})
                    dfs.append(sub)
        if dfs:
            combined = pd.concat(dfs, ignore_index=True).dropna().drop_duplicates().reset_index(drop=True)
            out_file = os.path.join(target_dir, filename)
            combined.to_csv(out_file, index=False, encoding='utf-8-sig')
            print(f"[OK] {filename}: Saved {len(combined)} clean deduplicated rows")

if __name__ == "__main__":
    setup_descriptive_bilingual_folder()
