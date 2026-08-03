import os
import shutil
import glob

def cleanup_redundant_files():
    print("=== Cleaning Redundant Files & Directories to Free PC Disk Space ===")
    
    workspace_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
    
    # Keep data_train_tringual in root
    keep_dir = os.path.join(workspace_dir, "data_train_tringual")
    
    # List redundant files/directories to delete
    redundant_paths = [
        # Redundant duplicate folder inside data/
        os.path.join(workspace_dir, "data", "data_train_tringual"),
        # Redundant large master duplicate copies
        os.path.join(workspace_dir, "data", "clean", "Master_Trilingual_Eng_Ekegusii_Swahili_34k.csv"),
        os.path.join(workspace_dir, "data", "clean", "Master_English_Ekegusii_Parallel_40k.csv"),
        os.path.join(workspace_dir, "data", "clean", "Master_Ekegusii_Swahili_Parallel_35k.csv"),
        os.path.join(workspace_dir, "data", "final_data", "Master_Multidomain_Trilingual_Dataset.csv"),
        os.path.join(workspace_dir, "data", "final_data", "Trilingual_Bible_Dataset.csv"),
    ]
    
    freed_bytes = 0
    
    for path in redundant_paths:
        if os.path.isdir(path):
            try:
                # count bytes before deleting
                for root, dirs, files in os.walk(path):
                    for f in files:
                        freed_bytes += os.path.getsize(os.path.join(root, f))
                shutil.rmtree(path)
                print(f"[REMOVED FOLDER] {path}")
            except Exception as e:
                print(f"[ERROR] Could not remove {path}: {e}")
        elif os.path.isfile(path):
            try:
                freed_bytes += os.path.getsize(path)
                os.remove(path)
                print(f"[DELETED FILE] {path}")
            except Exception as e:
                print(f"[ERROR] Could not remove {path}: {e}")
                
    freed_mb = freed_bytes / (1024 * 1024)
    print(f"\n[SUCCESS] Successfully freed {freed_mb:.2f} MB of disk space!")
    print(f"[ACTIVE FOLDER] Clean Training Folder: {keep_dir}")

if __name__ == "__main__":
    cleanup_redundant_files()
