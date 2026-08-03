import urllib.request
import json
import os
import pandas as pd

def download_quran_corpus():
    print("=== Downloading Parallel Quran Corpus (English - Swahili) ===")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # 1. Swahili translation (Sheikh Ali Muhsin Al-Barwani)
    print("Fetching Swahili translation (sw.barwani)...")
    sw_url = "https://api.alquran.cloud/v1/quran/sw.barwani"
    sw_req = urllib.request.Request(sw_url, headers=headers)
    sw_data = json.loads(urllib.request.urlopen(sw_req).read().decode('utf-8'))
    
    # 2. English translation (Sahih International)
    print("Fetching English translation (en.sahih)...")
    en_url = "https://api.alquran.cloud/v1/quran/en.sahih"
    en_req = urllib.request.Request(en_url, headers=headers)
    en_data = json.loads(urllib.request.urlopen(en_req).read().decode('utf-8'))
    
    sw_surahs = sw_data['data']['surahs']
    en_surahs = en_data['data']['surahs']
    
    rows = []
    
    for s_idx in range(len(sw_surahs)):
        sw_s = sw_surahs[s_idx]
        en_s = en_surahs[s_idx]
        
        surah_num = sw_s['number']
        surah_name_en = en_s['englishName']
        
        sw_ayahs = sw_s['ayahs']
        en_ayahs = en_s['ayahs']
        
        for a_idx in range(len(sw_ayahs)):
            sw_a = sw_ayahs[a_idx]
            en_a = en_ayahs[a_idx]
            
            rows.append({
                'english': en_a['text'].strip(),
                'swahili': sw_a['text'].strip()
            })
            
    df = pd.DataFrame(rows)
    print(f"Total Parallel Verses Extracted: {len(df)}")
    
    out_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, "English_Swahili_Parallel_Quran.csv")
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    download_quran_corpus()
