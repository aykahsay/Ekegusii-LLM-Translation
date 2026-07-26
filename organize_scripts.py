"""
organize_scripts.py
-------------------
Organizes all Python (.py) and Jupyter notebook (.ipynb) files into clean,
modular subdirectories under src/ and notebooks/.

Structure:
  src/
  ├── scrapers/          (PDF extractors & web scraping scripts)
  ├── data_processing/   (Preprocessing, verification & master dataset builders)
  ├── translation/       (Translation pipelines)
  └── training_eval/     (Model training, classifier & evaluation scripts)
  notebooks/             (Colab & Jupyter notebooks)
  app.py                 (Main Streamlit web entrypoint in root)
"""

import os, shutil

SCRAPER_FILES = [
    "data_collection_pipeline.py",
    "download_real_psa_data.py",
    "extract_pdf_to_csv.py",
    "extract_psas_from_pdfs.py",
    "extract_real_web_psas.py",
    "process_treasury_pdfs.py",
    "scrape_and_filter_psas.py",
    "scrape_and_translate_psas.py",
    "scrape_authentic_web_psas.py",
    "scrape_psas.py",
]

PROCESSING_FILES = [
    "build_final_datasets.py",
    "build_master_datasets.py",
    "check_all_psas.py",
    "check_true_psas.py",
    "generate_psa_metadata.py",
    "inspect_psa_columns.py",
    "organize_data_folders.py",
    "preprocess.py",
    "preprocess_ekegusii.py",
    "verify_scraped_psas.py",
]

TRANSLATION_FILES = [
    "translate_psas.py",
    "translate_dataset_to_ekegusii.py",
]

TRAINING_EVAL_FILES = [
    "train.py",
    "train_ekegusii.py",
    "train_psa_classifier.py",
    "evaluate.py",
    "evaluate_model.py",
    "retriever_setup.py",
]

NOTEBOOK_FILES = [
    "colab_training.ipynb",
    "create_colab_ekegusii.py",
    "create_colab_notebook.py",
    "kenya_treasury_scraper_and_translation.ipynb",
    "train_ekegusii_colab.ipynb",
]

MAPPING = [
    ("src/scrapers", SCRAPER_FILES),
    ("src/data_processing", PROCESSING_FILES),
    ("src/translation", TRANSLATION_FILES),
    ("src/training_eval", TRAINING_EVAL_FILES),
    ("notebooks", NOTEBOOK_FILES),
]

def organize():
    print("=" * 80)
    print("  ORGANIZING PYTHON AND NOTEBOOK SCRIPT STRUCTURE")
    print("=" * 80)

    for folder, file_list in MAPPING:
        os.makedirs(folder, exist_ok=True)
        for fname in file_list:
            if os.path.exists(fname):
                dst = os.path.join(folder, fname)
                shutil.move(fname, dst)
                print(f"  --> Moved {fname:<35} -> {folder}/")

    print("\n[DONE] Codebase scripts organized cleanly.")

if __name__ == "__main__":
    organize()
