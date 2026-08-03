import pandas as pd
import os
import glob

def audit_workspace_coverage():
    print("=== Auditing Entire Workspace Data Coverage vs 3 Active Training Folders ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    
    # 1. Collect all sentences across the 3 Training Folders
    trilingual_set = set()
    bilingual_set = set()
    unilingual_set = set()
    
    # Trilingual
    for f in glob.glob(os.path.join(workspace_dir, "data_train_tringual", "*.csv")):
        df = pd.read_csv(f)
        for col in ['English', 'Kiswahili', 'Ekegusii']:
            if col in df.columns:
                trilingual_set.update(df[col].astype(str).str.strip().str.lower())
                
    # Bilingual
    for f in glob.glob(os.path.join(workspace_dir, "data_train_bilingual", "*.csv")):
        df = pd.read_csv(f)
        for col in ['English', 'Kiswahili', 'Ekegusii', 'english', 'swahili', 'ekegusii']:
            if col in df.columns:
                bilingual_set.update(df[col].astype(str).str.strip().str.lower())

    # Unilingual
    for f in glob.glob(os.path.join(workspace_dir, "data_train_unilingual", "*.csv")):
        df = pd.read_csv(f)
        for col in ['English', 'Kiswahili', 'Ekegusii']:
            if col in df.columns:
                unilingual_set.update(df[col].astype(str).str.strip().str.lower())
                
    all_covered_sentences = trilingual_set | bilingual_set | unilingual_set
    print(f"Total Unique Sentences Covered in Training Folders: {len(all_covered_sentences)}")
    print(f"  - Trilingual Set: {len(trilingual_set)} unique sentences")
    print(f"  - Bilingual Set:  {len(bilingual_set)} unique sentences")
    print(f"  - Unilingual Set: {len(unilingual_set)} unique sentences")
    
    # 2. Audit all other workspace CSV files
    all_workspace_csvs = glob.glob(os.path.join(workspace_dir, "**", "*.csv"), recursive=True)
    
    training_folder_names = ["data_train_tringual", "data_train_bilingual", "data_train_unilingual"]
    
    uncovered_report = {}
    
    for csv_path in all_workspace_csvs:
        # Skip files inside training folders
        if any(tf in csv_path for tf in training_folder_names):
            continue
            
        fname = os.path.relpath(csv_path, workspace_dir)
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
                
            text_cols = [c for c in df.columns if any(k in c.lower() for k in ['english', 'kiswahili', 'swahili', 'ekegusii', 'text', 'row'])]
            if not text_cols:
                continue
                
            uncovered_count = 0
            total_rows = len(df)
            
            for _, row in df.iterrows():
                row_covered = False
                for c in text_cols:
                    val = str(row[c]).strip().lower()
                    if len(val) > 3 and val in all_covered_sentences:
                        row_covered = True
                        break
                if not row_covered and total_rows > 0:
                    uncovered_count += 1
                    
            if uncovered_count > 0:
                uncovered_report[fname] = {
                    'total_rows': total_rows,
                    'uncovered_rows': uncovered_count,
                    'columns': list(df.columns)
                }
                
        except Exception as e:
            print(f"Error auditing {fname}: {e}")
            
    print("\n=== AUDIT RESULTS: Workspace Files with Data NOT in Training Folders ===")
    if not uncovered_report:
        print("[ALL COVERED] Every single data row in your workspace is included in the 3 training folders!")
    else:
        for fname, info in uncovered_report.items():
            pct = (info['uncovered_rows'] / info['total_rows']) * 100
            print(f"- File: '{fname}' | Total Rows: {info['total_rows']} | Uncovered Rows: {info['uncovered_rows']} ({pct:.1f}%) | Columns: {info['columns']}")

if __name__ == "__main__":
    audit_workspace_coverage()
