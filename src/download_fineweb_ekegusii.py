import urllib.request
import json
import pandas as pd
import os
import time

def download_fineweb_ekegusii():
    print("=== Downloading Web Crawled Ekegusii Corpus (HuggingFaceFW/fineweb-2: guz_Latn) ===")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-2&config=guz_Latn&split=train"
    
    limit = 100
    offset = 0
    all_texts = []
    
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
                text = item.get('text', '').strip()
                url_src = item.get('url', '')
                if text:
                    all_texts.append({
                        'ekegusii_text': text,
                        'source_url': url_src
                    })
                    
            print(f"  - Downloaded offset {offset}..{offset+len(rows)} (Total pages extracted: {len(all_texts)})")
            offset += len(rows)
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            break
            
    if all_texts:
        df = pd.DataFrame(all_texts)
        out_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
        out_path = os.path.join(out_dir, "FineWeb_Ekegusii_Web_Corpus.csv")
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] Saved {len(df)} Web-Crawled Ekegusii Pages to {out_path}")

if __name__ == "__main__":
    download_fineweb_ekegusii()
