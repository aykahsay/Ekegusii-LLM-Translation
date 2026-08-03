import pandas as pd
import os
import re

def split_paragraph_to_sentences(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    # Split by newlines or sentence terminators
    lines = text.strip().split('\n')
    cleaned_lines = []
    for l in lines:
        l_str = l.strip()
        l_str = re.sub(r'^["\'`]+|["\'`]+$', '', l_str).strip()
        if l_str and len(l_str) > 3:
            cleaned_lines.append(l_str)
    return cleaned_lines

def process_rms_news_to_sentences():
    file_path = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean\Web_News_RMS_English_Ekegusii.csv"
    print(f"Processing {file_path} into sentence-by-sentence pairs...")
    
    df = pd.read_csv(file_path)
    print(f"Original paragraph blocks count: {len(df)}")
    
    sentence_pairs = []
    
    for idx, row in df.iterrows():
        eng_p = str(row['english']) if 'english' in row else ''
        guz_p = str(row['ekegusii']) if 'ekegusii' in row else ''
        
        eng_sentences = split_paragraph_to_sentences(eng_p)
        guz_sentences = split_paragraph_to_sentences(guz_p)
        
        # Align line by line if counts match or take minimum
        min_len = min(len(eng_sentences), len(guz_sentences))
        for i in range(min_len):
            e_s = eng_sentences[i]
            g_s = guz_sentences[i]
            if len(e_s) > 3 and len(g_s) > 3:
                sentence_pairs.append({
                    'english': e_s,
                    'ekegusii': g_s
                })
                
    sentence_df = pd.DataFrame(sentence_pairs).drop_duplicates().reset_index(drop=True)
    print(f"[OK] Extracted {len(sentence_df)} clean sentence-by-sentence parallel pairs!")
    
    # Save sentence-by-sentence CSV
    sentence_df.to_csv(file_path, index=False, encoding='utf-8-sig')
    
    out_sentences_path = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean\Web_News_RMS_English_Ekegusii_Sentences.csv"
    sentence_df.to_csv(out_sentences_path, index=False, encoding='utf-8-sig')
    print(f"[OK] Saved to: {file_path}")
    print(f"[OK] Saved to: {out_sentences_path}")

if __name__ == "__main__":
    process_rms_news_to_sentences()
