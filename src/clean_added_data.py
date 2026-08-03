import pandas as pd
import glob
import os
import re

def clean_text_field(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.strip()
    # Remove markdown header tokens
    text = re.sub(r'^#+\s*', '', text)
    # Remove metadata lines like * License: ... * Translation: ...
    if 'License:' in text or 'Illustration:' in text or 'Translation:' in text or 'Text:' in text:
        return ""
    # Remove excess quotes
    text = re.sub(r'^["\'`]+|["\'`]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_text(text):
    if not text or len(text) < 3:
        return False
    # Check if text is just metadata
    if any(k in text for k in ['License:', 'Illustration:', 'Translation:', 'CC-BY', 'Folktale']):
        return False
    return True

def clean_all_datasets():
    clean_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
    csv_files = glob.glob(os.path.join(clean_dir, "*.csv"))
    
    print(f"Cleaning {len(csv_files)} datasets in {clean_dir}...\n")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            orig_count = len(df)
            
            # 1. Drop metadata columns if present
            df = df.drop(columns=['book', 'chapter', 'verse_num', 'story_id', 'section_num'], errors='ignore')
            
            # 2. Clean text fields
            for col in df.columns:
                df[col] = df[col].astype(str).apply(clean_text_field)
                
            # 3. Filter valid rows (where all text columns have valid content)
            valid_mask = df.apply(lambda row: all(is_valid_text(str(row[c])) for c in df.columns), axis=1)
            df = df[valid_mask]
            
            # 4. Remove duplicates
            df = df.drop_duplicates().reset_index(drop=True)
            
            new_count = len(df)
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"[OK] Cleaned {filename}: {orig_count} -> {new_count} rows (removed {orig_count - new_count} noise/license rows)")
            
        except Exception as e:
            print(f"[ERROR] Error cleaning {filename}: {e}")

if __name__ == "__main__":
    clean_all_datasets()
