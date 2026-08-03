import os
import glob
import pandas as pd
import json

def parse_and_align_trilingual():
    base_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data"
    guz_dir = os.path.join(base_dir, "bible_ekegusii", "guz_readaloud")
    eng_dir = os.path.join(base_dir, "bible_ekegusii", "eng-web_readaloud")
    swh_dir = os.path.join(base_dir, "bible_swahili", "swhonen_readaloud")
    
    guz_files = glob.glob(os.path.join(guz_dir, "guz_*.txt"))
    
    trilingual_data = []
    eng_swh_data = []
    guz_swh_data = []
    
    for guz_path in sorted(guz_files):
        filename = os.path.basename(guz_path)
        if filename == "guz_000_000_000_read.txt":
            continue
            
        eng_filename = filename.replace("guz_", "eng-web_")
        swh_filename = filename.replace("guz_", "swhonen_")
        
        eng_path = os.path.join(eng_dir, eng_filename)
        swh_path = os.path.join(swh_dir, swh_filename)
        
        has_eng = os.path.exists(eng_path)
        has_swh = os.path.exists(swh_path)
        
        with open(guz_path, 'r', encoding='utf-8') as fg:
            guz_lines = [line.strip() for line in fg.readlines() if line.strip()]
            
        eng_lines = []
        if has_eng:
            with open(eng_path, 'r', encoding='utf-8') as fe:
                eng_lines = [line.strip() for line in fe.readlines() if line.strip()]
                
        swh_lines = []
        if has_swh:
            with open(swh_path, 'r', encoding='utf-8') as fs:
                swh_lines = [line.strip() for line in fs.readlines() if line.strip()]
                
        parts = filename.split('_')
        book_code = parts[2] if len(parts) > 3 else "UNKNOWN"
        chapter = parts[3] if len(parts) > 3 else "1"
        
        guz_verses = guz_lines[2:]
        eng_verses = eng_lines[2:] if has_eng else []
        swh_verses = swh_lines[2:] if has_swh else []
        
        # Determine minimum lengths for pair matching
        if has_eng and has_swh:
            min_len = min(len(guz_verses), len(eng_verses), len(swh_verses))
            for idx in range(min_len):
                trilingual_data.append({
                    "book": book_code,
                    "chapter": int(chapter) if chapter.isdigit() else chapter,
                    "verse_num": idx + 1,
                    "english": eng_verses[idx],
                    "ekegusii": guz_verses[idx],
                    "swahili": swh_verses[idx]
                })
                
        if has_swh and has_eng:
            min_len = min(len(eng_verses), len(swh_verses))
            for idx in range(min_len):
                eng_swh_data.append({
                    "book": book_code,
                    "chapter": int(chapter) if chapter.isdigit() else chapter,
                    "verse_num": idx + 1,
                    "english": eng_verses[idx],
                    "swahili": swh_verses[idx]
                })

        if has_swh:
            min_len = min(len(guz_verses), len(swh_verses))
            for idx in range(min_len):
                guz_swh_data.append({
                    "book": book_code,
                    "chapter": int(chapter) if chapter.isdigit() else chapter,
                    "verse_num": idx + 1,
                    "ekegusii": guz_verses[idx],
                    "swahili": swh_verses[idx]
                })

    df_tri = pd.DataFrame(trilingual_data)
    df_eng_swh = pd.DataFrame(eng_swh_data)
    df_guz_swh = pd.DataFrame(guz_swh_data)
    
    clean_dir = os.path.join(base_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)
    
    tri_csv = os.path.join(clean_dir, "Trilingual_English_Ekegusii_Swahili_Parallel_Bible.csv")
    eng_swh_csv = os.path.join(clean_dir, "English_Swahili_Parallel_Bible.csv")
    guz_swh_csv = os.path.join(clean_dir, "Ekegusii_Swahili_Parallel_Bible.csv")
    
    df_tri.to_csv(tri_csv, index=False, encoding='utf-8-sig')
    df_eng_swh.to_csv(eng_swh_csv, index=False, encoding='utf-8-sig')
    df_guz_swh.to_csv(guz_swh_csv, index=False, encoding='utf-8-sig')
    
    print("=== Alignment Complete ===")
    print(f"Trilingual (English - Ekegusii - Swahili): {len(df_tri)} aligned sentence rows.")
    print(f"English - Swahili: {len(df_eng_swh)} aligned sentence rows.")
    print(f"Ekegusii - Swahili: {len(df_guz_swh)} aligned sentence rows.")
    print(f"Saved to: {tri_csv}")

if __name__ == "__main__":
    parse_and_align_trilingual()
