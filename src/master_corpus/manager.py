"""
Master Corpus Manager API
-------------------------
Centralized API for accessing master sentence and lexical corpora with 0% leakage guarantee.
"""

import os
import pandas as pd

class MasterCorpusManager:
    def __init__(self, data_dir="data/master_corpus"):
        self.data_dir = data_dir
        self.sentence_corpus_path = os.path.join(data_dir, "master_sentence_corpus.csv")
        self.lexical_corpus_path = os.path.join(data_dir, "master_lexical_corpus.csv")
        self.splits_dir = os.path.join(data_dir, "splits")
        
    def get_train_split(self):
        return pd.read_csv(os.path.join(self.splits_dir, "master_train.csv"))
        
    def get_val_split(self):
        return pd.read_csv(os.path.join(self.splits_dir, "master_val.csv"))
        
    def get_test_split(self):
        return pd.read_csv(os.path.join(self.splits_dir, "master_test.csv"))
        
    def verify_integrity(self):
        train = self.get_train_split()
        val = self.get_val_split()
        test = self.get_test_split()
        
        train_ids = set(train['concept_id'])
        val_ids = set(val['concept_id'])
        test_ids = set(test['concept_id'])
        
        overlap = (train_ids & test_ids) | (train_ids & val_ids) | (val_ids & test_ids)
        assert len(overlap) == 0, f"Data leakage detected! Overlap: {len(overlap)}"
        return True
