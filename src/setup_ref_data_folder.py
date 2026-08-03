import pandas as pd
import os
import shutil

def setup_ref_data_folder():
    print("=== Creating ref_data Folder & Copying Reference Datasets ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    ref_dir = os.path.join(workspace_dir, "ref_data")
    clean_dir = os.path.join(workspace_dir, "data", "clean")
    raw_dir = os.path.join(workspace_dir, "data", "raw")
    root_data_dir = os.path.join(workspace_dir, "data")
    
    os.makedirs(ref_dir, exist_ok=True)
    
    # List of reference sources and their target descriptive names in ref_data
    reference_sources = [
        (os.path.join(raw_dir, "reliefweb_kenya_psas.csv"), "ReliefWeb_Kenya_Disaster_PSAs_Raw.csv"),
        (os.path.join(clean_dir, "FineWeb_Ekegusii_Web_Corpus.csv"), "FineWeb_Ekegusii_Web_Corpus.csv"),
        (os.path.join(clean_dir, "Web_News_RMS_English_Ekegusii_Sentences.csv"), "Web_News_RMS_EgesaFM_English_Ekegusii.csv"),
        (os.path.join(clean_dir, "African_Storybooks_Trilingual.csv"), "African_Storybooks_Multilingual_Corpus.csv"),
        (os.path.join(clean_dir, "NDMA_English_Only.csv"), "NDMA_Drought_Advisories_English.csv"),
        (os.path.join(clean_dir, "Kiswahili_Ekegusii_Dictionary.csv"), "Structured_Swahili_Ekegusii_Dictionary.csv"),
        (os.path.join(clean_dir, "Online_Glosbe_Swahili_Ekegusii_Dictionary.csv"), "Online_Glosbe_Swahili_Ekegusii_Dictionary.csv"),
        (os.path.join(root_data_dir, "human_eval_100_sentences.csv"), "Human_Evaluation_Benchmark_100.csv"),
        (os.path.join(raw_dir, "scraped_psas_verified.csv"), "Scraped_Government_PSAs_Verified.csv"),
        (os.path.join(raw_dir, "moh_x_posts_translated.csv"), "Ministry_of_Health_Social_Posts.csv")
    ]
    
    copied_count = 0
    for src_path, dest_name in reference_sources:
        if os.path.exists(src_path):
            dest_path = os.path.join(ref_dir, dest_name)
            try:
                shutil.copy2(src_path, dest_path)
                df = pd.read_csv(dest_path)
                print(f"[OK] Copied '{dest_name}' ({len(df)} rows) -> {dest_path}")
                copied_count += 1
            except Exception as e:
                print(f"[ERROR] Could not copy {src_path}: {e}")
        else:
            print(f"[NOTE] Source file not found: {src_path}")
            
    print(f"\n[SUCCESS] Compiled {copied_count} reference datasets into '{ref_dir}'")

if __name__ == "__main__":
    setup_ref_data_folder()
