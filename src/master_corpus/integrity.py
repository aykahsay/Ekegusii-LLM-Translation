"""
Data Leakage & Integrity Checker
"""
from src.master_corpus.manager import MasterCorpusManager

if __name__ == "__main__":
    manager = MasterCorpusManager()
    if manager.verify_integrity():
        print("[SUCCESS] Master Corpus 0% Data Leakage Verified!")
