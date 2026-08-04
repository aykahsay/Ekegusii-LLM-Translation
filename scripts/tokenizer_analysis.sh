#!/bin/bash
# ==============================================================================
# Qwen2.5 vs. Mistral-7B Tokenizer Fertility & Vocabulary Coverage Analysis
# Backs notebook 04 (tokenizer_analysis) -- both Qwen/Qwen2.5-7B-Instruct and
# mistralai/Mistral-7B-Instruct-v0.3 are fully open on HuggingFace Hub, so no
# Hub access requests are needed.
# ==============================================================================

set -e

echo "======================================================================"
echo "Tokenizer Fertility & Vocabulary Coverage Analysis"
echo "======================================================================"

python -c "
from src.master_corpus.manager import MasterCorpusManager
from src.tokenizer.compare import TokenizerComparator
from src.utils.constants import SUPPORTED_LANGUAGES

manager = MasterCorpusManager()
df = manager.load_sentence_corpus()

language_sentences = {
    lang: df[lang].dropna().astype(str).tolist()
    for lang in SUPPORTED_LANGUAGES
}

comparator = TokenizerComparator()
fertility_df = comparator.compare(language_sentences)
print(fertility_df)
print()

coverage_df = comparator.compare_vocabulary_coverage(language_sentences)
print(coverage_df)
print()

recommendation = comparator.recommend_base_model(fertility_df, target_language='Ekegusii')
print(f'Recommended base model for Ekegusii (lower fertility): {recommendation}')
"

echo "======================================================================"
echo "[SUCCESS] Tokenizer analysis complete."
echo "======================================================================"
