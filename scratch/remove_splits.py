import os
import shutil

print("=== Removing Train / Validation / Test Splits Directory ===")

base_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
splits_dir = os.path.join(base_dir, "dataset", "splits")

if os.path.exists(splits_dir):
    shutil.rmtree(splits_dir)
    print(f"Successfully deleted directory: {splits_dir}")
else:
    print(f"Directory {splits_dir} does not exist.")
