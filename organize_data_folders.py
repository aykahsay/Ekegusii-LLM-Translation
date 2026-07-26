"""
organize_data_folders.py
------------------------
Organizes all project data files into a clean folder structure inside data/:

  data/
  ├── Master_Mixed_Data.csv      (Master clean dataset - Mixed)
  ├── Master_PSA_Only.csv        (Master clean dataset - PSA only)
  ├── raw/                       (All raw scraped CSVs, web outputs & PDFs)
  │   ├── pdfs/
  │   ├── web_scraped/
  │   └── notice_directories/
  └── intermediate/              (Verification outputs, legacy splits & temporary files)
"""

import sys, io, os, shutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")

def main():
    print("=" * 80)
    print("  ORGANIZING DATA DIRECTORY STRUCTURE")
    print("=" * 80)

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    raw_files = [
        "treasury_psas.csv",
        "extracted_pdf_psas.csv",
        "scraped_psas_english.csv",
        "scraped_psas_verified.csv",
        "scraped_psas_translated.csv",
        "scraped_psas_part4.csv",
        "scraped_psas_part3.csv",
        "PSA.csv",
        "Scrapped_psas.xlsx",
        "real_web_psas.csv",
        "kenya_bulk_notices_extracted.csv",
        "extracted_pdf_data.csv",
        "moh_x_posts_translated.csv",
        "psa_campaigns.json",
        "high_quality_scraped_psas.csv"
    ]

    intermediate_files = [
        "all_psas_verified.csv",
        "confirmed_psas.csv",
        "non_psas.csv",
        "Mixed_data.csv",
        "PSA_only.csv",
        "train.csv",
        "test.csv",
        "dev.csv",
        "train_guz.csv",
        "test_guz.csv",
        "dev_guz.csv",
        "_PSA_EnGuz.csv",
        "train_test_combined.csv",
        "train_test_guz_combined.csv"
    ]

    print("\n📁 Moving raw scraped files to data/raw/...")
    for fname in raw_files:
        src = os.path.join(DATA_DIR, fname)
        if not os.path.exists(src) and os.path.exists(fname):
            src = fname
        if os.path.exists(src):
            dst = os.path.join(RAW_DIR, os.path.basename(fname))
            shutil.move(src, dst)
            print(f"  --> Moved {fname} to data/raw/")

    for fname in os.listdir(DATA_DIR):
        src = os.path.join(DATA_DIR, fname)
        if fname in ["raw", "intermediate", "Master_Mixed_Data.csv", "Master_PSA_Only.csv"]:
            continue
        
        if fname.endswith(".pdf"):
            pdf_dir = os.path.join(RAW_DIR, "pdfs")
            os.makedirs(pdf_dir, exist_ok=True)
            dst = os.path.join(pdf_dir, fname)
            shutil.move(src, dst)
            print(f"  --> Moved PDF {fname} to data/raw/pdfs/")
        elif os.path.isdir(src) and fname not in ["raw", "intermediate"]:
            dst = os.path.join(RAW_DIR, fname)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            print(f"  --> Moved directory {fname} to data/raw/")

    print("\n📁 Moving intermediate processing files to data/intermediate/...")
    for fname in intermediate_files:
        src = os.path.join(DATA_DIR, fname)
        if os.path.exists(src):
            dst = os.path.join(INTERMEDIATE_DIR, fname)
            shutil.move(src, dst)
            print(f"  --> Moved {fname} to data/intermediate/")

    print("\n[DONE] Directory structure organized.")

if __name__ == "__main__":
    main()
