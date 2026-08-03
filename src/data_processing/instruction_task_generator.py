"""
Module 2: Multilingual Instruction Task Generator
---------------------------------------------------
Transforms rows from the Master Sentence Corpus into structured, bidirectional 
instruction-tuning examples for LLMs (Aya 23, Llama 3.1, NLLB).

Generates up to 6 translation tasks per concept row:
 1. English -> Ekegusii
 2. Ekegusii -> English
 3. Kiswahili -> Ekegusii
 4. Ekegusii -> Kiswahili
 5. English -> Kiswahili
 6. Kiswahili -> English
"""

import os
import json
import pandas as pd
import numpy as np

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
MASTER_SPLITS_DIR = os.path.join(WORKSPACE_DIR, "data", "master_corpus", "splits")
INSTRUCTION_DIR = os.path.join(WORKSPACE_DIR, "data", "instruction_datasets")

os.makedirs(INSTRUCTION_DIR, exist_ok=True)

PROMPT_TEMPLATES = {
    ('English', 'Ekegusii'): "Translate the following English text into Ekegusii:\n\nInput: {src}\n\nOutput:",
    ('Ekegusii', 'English'): "Translate the following Ekegusii text into English:\n\nInput: {src}\n\nOutput:",
    ('Kiswahili', 'Ekegusii'): "Translate the following Kiswahili text into Ekegusii:\n\nInput: {src}\n\nOutput:",
    ('Ekegusii', 'Kiswahili'): "Translate the following Ekegusii text into Kiswahili:\n\nInput: {src}\n\nOutput:",
    ('English', 'Kiswahili'): "Translate the following English text into Kiswahili:\n\nInput: {src}\n\nOutput:",
    ('Kiswahili', 'English'): "Translate the following Kiswahili text into English:\n\nInput: {src}\n\nOutput:"
}

def generate_instruction_pairs_for_row(row):
    tasks = []
    concept_id = row['concept_id']
    eng = str(row['English']).strip() if pd.notna(row.get('English')) else ""
    swa = str(row['Kiswahili']).strip() if pd.notna(row.get('Kiswahili')) else ""
    eke = str(row['Ekegusii']).strip() if pd.notna(row.get('Ekegusii')) else ""
    
    pairs = [
        ('English', 'Ekegusii', eng, eke),
        ('Ekegusii', 'English', eke, eng),
        ('Kiswahili', 'Ekegusii', swa, eke),
        ('Ekegusii', 'Kiswahili', eke, swa),
        ('English', 'Kiswahili', eng, swa),
        ('Kiswahili', 'English', swa, eng)
    ]
    
    for src_l, tgt_l, src_txt, tgt_txt in pairs:
        if src_txt and tgt_txt and len(src_txt) > 1 and len(tgt_txt) > 1:
            template = PROMPT_TEMPLATES[(src_l, tgt_l)]
            prompt = template.format(src=src_txt)
            tasks.append({
                'concept_id': concept_id,
                'src_lang': src_l,
                'tgt_lang': tgt_l,
                'prompt': prompt,
                'input': src_txt,
                'output': tgt_txt,
                'source': row.get('source', 'Unknown')
            })
            
    return tasks

def generate_all_instruction_datasets():
    print("=== Module 2: Generating Multilingual Instruction Tuning Tasks ===")
    
    for split_name in ['master_train', 'master_val', 'master_test']:
        fpath = os.path.join(MASTER_SPLITS_DIR, f"{split_name}.csv")
        if not os.path.exists(fpath):
            print(f"Skipping {split_name}, file not found: {fpath}")
            continue
            
        df = pd.read_csv(fpath)
        all_instructions = []
        
        for idx, row in df.iterrows():
            tasks = generate_instruction_pairs_for_row(row)
            all_instructions.extend(tasks)
            
        instr_df = pd.DataFrame(all_instructions)
        out_csv = os.path.join(INSTRUCTION_DIR, f"instruction_{split_name}.csv")
        out_jsonl = os.path.join(INSTRUCTION_DIR, f"instruction_{split_name}.jsonl")
        
        instr_df.to_csv(out_csv, index=False)
        instr_df.to_json(out_jsonl, orient='records', lines=True)
        
        print(f" [OK] {split_name}: {len(df)} concepts -> {len(instr_df)} instruction-tuning tasks generated!")
        print(f"      Saved to: '{out_jsonl}'")

if __name__ == "__main__":
    generate_all_instruction_datasets()
