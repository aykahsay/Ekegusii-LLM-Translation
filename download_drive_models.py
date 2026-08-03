"""
download_drive_models.py
-------------------------
Utility script to download trained fine-tuned model checkpoints from Google Drive into local 'models/' directory.
"""

import os, sys, io, subprocess

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DRIVE_MODELS_URL = "https://drive.google.com/drive/folders/1fYLEQfIVKR4e5zi_F9plAjazJ-RcIfcC"
LOCAL_MODELS_DIR = "models"

def main():
    print("=" * 80)
    print("  DOWNLOADING TRAINED FINE-TUNED MODELS FROM GOOGLE DRIVE")
    print("=" * 80)
    
    os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)
    
    try:
        import gdown
    except ImportError:
        print("[*] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    print(f"[*] Downloading trained model folder from: {DRIVE_MODELS_URL}")
    print(f"[*] Saving into: {os.path.abspath(LOCAL_MODELS_DIR)}")
    
    try:
        gdown.download_folder(url=DRIVE_MODELS_URL, output=LOCAL_MODELS_DIR, quiet=False)
        print("\n[SUCCESS] Download completed successfully!")
        print("💡 Restart or refresh Streamlit (http://localhost:8501) to use YOUR fine-tuned models!")
    except Exception as e:
        print(f"\n[WARNING] Automated download encountered an issue: {e}")
        print("\n👉 Manual Download Instructions:")
        print(f"   1. Open Google Drive in browser: {DRIVE_MODELS_URL}")
        print("   2. Download the trained model folders")
        print(f"   3. Extract them directly into: {os.path.abspath(LOCAL_MODELS_DIR)}")

if __name__ == "__main__":
    main()
