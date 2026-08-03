import urllib.request
import json
import pandas as pd
import os
import time

def download_hf_finetranslations():
    print("=== Downloading Web News Parallel Corpus from HuggingFace (finetranslations: guz_Latn) ===")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/finetranslations&config=guz_Latn&split=train"
    
    limit = 100
    offset = 0
    all_rows = []
    
    while True:
        url = f"{base_url}&offset={offset}&limit={limit}"
        try:
            req = urllib.request.Request(url, headers=headers)
            res = urllib.request.urlopen(req).read().decode('utf-8')
            data = json.loads(res)
            
            rows = data.get('rows', [])
            if not rows:
                print(f"No more rows found at offset {offset}.")
                break
                
            for r in rows:
                item = r.get('row', {})
                guz_text = item.get('og_full_text', '').strip()
                eng_text = item.get('translated_text', '').strip()
                url_src = item.get('url', '')
                
                if guz_text and eng_text:
                    all_rows.append({
                        'english': eng_text,
                        'ekegusii': guz_text,
                        'source_url': url_src
                    })
                    
            print(f"  - Downloaded offset {offset}..{offset+len(rows)} (Total extracted: {len(all_rows)})")
            offset += len(rows)
            
            # rate limit protection
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break
            
    if all_rows:
        df = pd.DataFrame(all_rows)
        out_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
        out_path = os.path.join(out_dir, "Web_News_RMS_English_Ekegusii.csv")
        df[['english', 'ekegusii']].to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] Saved {len(df)} Web News English-Ekegusii Parallel Rows to {out_path}")

if __name__ == "__main__":
    download_hf_finetranslations()
