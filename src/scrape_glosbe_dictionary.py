import urllib.request
import re
import json
import pandas as pd
import os
import time

def scrape_glosbe_sw_guz():
    print("=== Scraping Online Swahili - Ekegusii Dictionary & Translation Memory from Glosbe ===")
    
    # Common Swahili terms to query on Glosbe
    sw_words = [
        "habari", "jambo", "rafiki", "mtu", "watu", "mume", "mke", "mtoto", "watoto", "baba", "mama",
        "kaka", "dada", "mwalimu", "mzee", "kijana", "nyumba", "maji", "moto", "mti", "ardhi", "nchi",
        "jua", "mwezi", "mvua", "chakula", "ugali", "mboga", "nyama", "maziwa", "ndizi", "shamba",
        "mbolea", "afya", "ugonjwa", "hospitali", "dawa", "kichwa", "mkono", "mguu", "jicho", "tumbo",
        "moyo", "kula", "kunywa", "kulima", "kusoma", "kusema", "kutembea", "kuja", "kwenda", "kuona",
        "kujua", "kufanya", "kununua", "kuuza", "siku", "mwaka", "usiku", "asubuhi", "jioni", "moja",
        "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa", "kumi", "fedha", "pesa", "amani"
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    dictionary_records = []
    
    for word in sw_words:
        url = f"https://glosbe.com/sw/guz/{word}"
        try:
            req = urllib.request.Request(url, headers=headers)
            html = urllib.request.urlopen(req).read().decode('utf-8', errors='ignore')
            
            # Find translation matches in Glosbe HTML
            matches = re.findall(r'<h3[^>]*class="[^"]*translation[^"]*"[^>]*>\s*([^<]+)\s*</h3>', html, re.IGNORECASE)
            if not matches:
                matches = re.findall(r'lang="guz"[^>]*>\s*([^<]+)\s*<', html, re.IGNORECASE)
                
            clean_matches = list(set([m.strip() for m in matches if m.strip() and len(m.strip()) < 40 and not m.strip().startswith('{')]))
            
            if clean_matches:
                guz_trans = ", ".join(clean_matches[:3])
                dictionary_records.append({
                    'swahili': word,
                    'ekegusii': guz_trans,
                    'source': 'Glosbe Online Dictionary'
                })
                print(f"  - [Found] Swahili: '{word}' -> Ekegusii: '{guz_trans}'")
                
            time.sleep(0.3)
            
        except Exception as e:
            print(f"Error scraping '{word}': {e}")
            
    if dictionary_records:
        df = pd.DataFrame(dictionary_records)
        out_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
        out_path = os.path.join(out_dir, "Online_Glosbe_Swahili_Ekegusii_Dictionary.csv")
        df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] Saved {len(df)} Online Glosbe Swahili-Ekegusii Dictionary Entries to {out_path}")

if __name__ == "__main__":
    scrape_glosbe_sw_guz()
