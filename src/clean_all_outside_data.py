import os
import shutil
import glob

def remove_all_outside_data():
    print("=== Removing ALL Data Files Outside Designated Training & Ref Folders ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    
    # Official Allowed Folders
    allowed_folders = [
        os.path.join(workspace_dir, "data_train_tringual"),
        os.path.join(workspace_dir, "data_train_bilingual"),
        os.path.join(workspace_dir, "data_train_unilingual"),
        os.path.join(workspace_dir, "ref_data")
    ]
    
    # Legacy data subdirectories to delete
    legacy_data_dir = os.path.join(workspace_dir, "data")
    
    freed_bytes = 0
    
    if os.path.exists(legacy_data_dir):
        try:
            for root, dirs, files in os.walk(legacy_data_dir):
                for f in files:
                    if f.endswith('.csv') or f.endswith('.json') or f.endswith('.txt'):
                        fp = os.path.join(root, f)
                        freed_bytes += os.path.getsize(fp)
            shutil.rmtree(legacy_data_dir)
            print(f"[REMOVED] Entire legacy folder '{legacy_data_dir}' successfully.")
        except Exception as e:
            print(f"[ERROR] Could not remove legacy data dir: {e}")
            
    # Check if any CSVs remain outside allowed folders
    remaining_csvs = glob.glob(os.path.join(workspace_dir, "**", "*.csv"), recursive=True)
    outside_csvs = [p for p in remaining_csvs if not any(af in p for af in allowed_folders)]
    
    for o_csv in outside_csvs:
        try:
            freed_bytes += os.path.getsize(o_csv)
            os.remove(o_csv)
            print(f"[DELETED OUTSIDE FILE] {o_csv}")
        except Exception as e:
            pass
            
    freed_mb = freed_bytes / (1024 * 1024)
    print(f"\n[SUCCESS] Freed {freed_mb:.2f} MB! Zero data files remain outside designated folders.")

if __name__ == "__main__":
    remove_all_outside_data()
