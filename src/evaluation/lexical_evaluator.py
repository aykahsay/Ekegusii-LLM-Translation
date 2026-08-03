"""
Lexical Benchmark Evaluator for Low-Resource NMT
--------------------------------------------------
Evaluates model translation performance on rare-word and domain-specific terminology 
derived from the Master Lexical Corpus.

Metrics computed:
1. Lexical Term Accuracy (%) - Exact dictionary term match in generated output.
2. Synonymous / Morphological Coverage (%) - Stemmed / sub-word match.
"""

import os
import re
import pandas as pd
import numpy as np

def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def evaluate_lexical_precision(predictions, references):
    """
    Args:
        predictions (list of str): Model output translations.
        references (list of str): Target dictionary term references (Ekegusii/English).
    
    Returns:
        dict: Lexical precision metrics.
    """
    exact_matches = 0
    partial_matches = 0
    total = len(references)
    
    for pred, ref in zip(predictions, references):
        pred_clean = normalize_text(pred)
        ref_clean = normalize_text(ref)
        
        if not ref_clean:
            continue
            
        # Check if reference term is in output
        if ref_clean in pred_clean:
            exact_matches += 1
            partial_matches += 1
        else:
            # Check for sub-word stem match (morphological prefix/suffix)
            ref_words = ref_clean.split()
            if any(w[:4] in pred_clean for w in ref_words if len(w) >= 4):
                partial_matches += 1
                
    exact_acc = (exact_matches / total * 100.0) if total > 0 else 0.0
    partial_acc = (partial_matches / total * 100.0) if total > 0 else 0.0
    
    return {
        'total_terms': total,
        'exact_lexical_accuracy': round(exact_acc, 2),
        'morphological_lexical_accuracy': round(partial_acc, 2)
    }

if __name__ == "__main__":
    # Test evaluation with sample data
    preds = ["eserekari ekobwatania ebiama", "kebe nomoroberio bwomorero esukuru"]
    refs = ["eserikari", "omorero"]
    results = evaluate_lexical_precision(preds, refs)
    print("Lexical Evaluator Test Results:", results)
