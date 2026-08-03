import os
import glob
import pandas as pd
import json

def parse_and_align():
    base_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\bible_ekegusii"
    guz_dir = os.path.join(base_dir, "guz_readaloud")
    eng_dir = os.path.join(base_dir, "eng-web_readaloud")
    
    guz_files = glob.glob(os.path.join(guz_dir, "guz_*.txt"))
    
    aligned_data = []
    skipped_files = []
    
    for guz_path in sorted(guz_files):
        filename = os.path.basename(guz_path)
        if filename == "guz_000_000_000_read.txt":
            continue
            
        eng_filename = filename.replace("guz_", "eng-web_")
        eng_path = os.path.join(eng_dir, eng_filename)
        
        if not os.path.exists(eng_path):
            skipped_files.append((filename, "English file missing"))
            continue
            
        with open(guz_path, 'r', encoding='utf-8') as fg:
            guz_lines = [line.strip() for line in fg.readlines() if line.strip()]
            
        with open(eng_path, 'r', encoding='utf-8') as fe:
            eng_lines = [line.strip() for line in fe.readlines() if line.strip()]
            
        # Parse book code and chapter
        # Filename format: guz_002_GEN_01_read.txt -> book = GEN, chapter = 01
        parts = filename.split('_')
        book_code = parts[2] if len(parts) > 3 else "UNKNOWN"
        chapter = parts[3] if len(parts) > 3 else "1"
        
        # Skip header lines (Line 0 is Book title, Line 1 is Chapter X)
        guz_verses = guz_lines[2:]
        eng_verses = eng_lines[2:]
        
        # Pair up to the minimum line length of matching verses
        min_len = min(len(guz_verses), len(eng_verses))
        for idx in range(min_len):
            g_text = guz_verses[idx]
            e_text = eng_verses[idx]
            
            aligned_data.append({
                "book": book_code,
                "chapter": int(chapter) if chapter.isdigit() else chapter,
                "verse_num": idx + 1,
                "english": e_text,
                "ekegusii": g_text
            })

    df = pd.DataFrame(aligned_data)
    
    csv_path = os.path.join(base_dir, "English_Ekegusii_Parallel_Bible.csv")
    jsonl_path = os.path.join(base_dir, "English_Ekegusii_Parallel_Bible.jsonl")
    
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    with open(jsonl_path, 'w', encoding='utf-8') as fj:
        for item in aligned_data:
            fj.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    # Also save to main data/languages/ or data/clean/ folder for easy access in project
    project_data_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
    os.makedirs(project_data_dir, exist_ok=True)
    df.to_csv(os.path.join(project_data_dir, "English_Ekegusii_Parallel_Bible.csv"), index=False, encoding='utf-8-sig')

    print(f"Successfully aligned {len(df)} sentence pairs across {len(guz_files)} chapter files!")
    print(f"Saved dataset to:")
    print(f" - {csv_path}")
    print(f" - {jsonl_path}")
    print(f" - {os.path.join(project_data_dir, 'English_Ekegusii_Parallel_Bible.csv')}")

if __name__ == "__main__":
    parse_and_align()
