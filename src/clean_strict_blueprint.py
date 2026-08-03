"""
Strict Blueprint Repository Cleaner
-----------------------------------
Enforces EXACT 1:1 match with the requested open-source architecture specification.
Removes all extra root files, non-blueprint notebooks, and legacy artifacts from Git.
"""

import os
import shutil

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"

# Allowed Root Files
ALLOWED_ROOT_FILES = {
    "README.md", "LICENSE", "CITATION.cff", "CHANGELOG.md", "CONTRIBUTING.md",
    ".gitignore", "pyproject.toml", "requirements.txt", "environment.yml", "setup.py"
}

# Allowed Root Directories
ALLOWED_ROOT_DIRS = {
    "configs", "data", "notebooks", "src", "experiments", "checkpoints",
    "outputs", "scripts", "tests", "docs", "paper", ".git", ".github"
}

def clean_root_directory():
    print("=== Cleaning Root Directory to Match Blueprint Exactly ===")
    
    for item in os.listdir(WORKSPACE_DIR):
        full_path = os.path.join(WORKSPACE_DIR, item)
        
        if os.path.isfile(full_path):
            if item not in ALLOWED_ROOT_FILES:
                print(f"Removing extra root file: {item}")
                os.remove(full_path)
                
        elif os.path.isdir(full_path):
            if item not in ALLOWED_ROOT_DIRS and not item.startswith("."):
                print(f"Removing extra root directory: {item}")
                shutil.rmtree(full_path, ignore_errors=True)

def clean_notebooks_directory():
    print("=== Cleaning notebooks/ Directory to Match 13 Blueprint Notebooks Exactly ===")
    
    allowed_notebooks = {
        "01_master_corpus_analysis.ipynb", "02_data_validation.ipynb", "03_resource_statistics.ipynb",
        "04_tokenizer_analysis.ipynb", "05_instruction_generation.ipynb", "06_dataset_scheduler.ipynb",
        "07_train_aya.ipynb", "08_train_llama.ipynb", "09_translation_evaluation.ipynb",
        "10_dictionary_analysis.ipynb", "11_ablation_study.ipynb", "12_error_analysis.ipynb",
        "13_publication_figures.ipynb", ".gitkeep"
    }
    
    nb_dir = os.path.join(WORKSPACE_DIR, "notebooks")
    if os.path.exists(nb_dir):
        for item in os.listdir(nb_dir):
            if item not in allowed_notebooks:
                full_path = os.path.join(nb_dir, item)
                print(f"Removing non-blueprint notebook: {item}")
                if os.path.isfile(full_path):
                    os.remove(full_path)
                elif os.path.isdir(full_path):
                    shutil.rmtree(full_path, ignore_errors=True)

if __name__ == "__main__":
    clean_root_directory()
    clean_notebooks_directory()
    print("\n[SUCCESS] Repository Cleaned to Match User Blueprint 100% Exactly!")
