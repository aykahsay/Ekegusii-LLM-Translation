#!/bin/bash
# ==============================================================================
# Master Corpus Build & Verification
# Loads the master sentence/lexical corpora, runs the zero-leakage audit,
# and prints corpus/split statistics -- run this first, before any
# task-generation or training script, to confirm the data is intact.
# ==============================================================================

set -e

echo "======================================================================"
echo "Master Corpus Build & Verification"
echo "======================================================================"

echo "[1/2] Verifying zero-leakage across train/val/test splits..."
python -m src.cli.main audit

echo "[2/2] Printing corpus statistics..."
python -c "
from src.master_corpus.manager import MasterCorpusManager
from src.master_corpus.statistics import CorpusStatistics

manager = MasterCorpusManager()
stats = CorpusStatistics()

sentence_df = manager.load_sentence_corpus()
lexical_df = manager.load_lexical_corpus()
splits = {
    'train': manager.load_train_split(),
    'val': manager.load_val_split(),
    'test': manager.load_test_split(),
}

print(f'Master Sentence Corpus: {len(sentence_df):,} concepts')
print(f'Master Lexical Corpus:  {len(lexical_df):,} entries')
print()
print(stats.split_summary(splits))
print()
print(stats.source_distribution(sentence_df, 'source'))
"

echo "======================================================================"
echo "[SUCCESS] Master corpus verified and statistics printed."
echo "======================================================================"
