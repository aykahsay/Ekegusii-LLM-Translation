import urllib.request
import json
import pandas as pd
import os

def search_and_download_hf_guz():
    print("=== Searching Hugging Face Hub for Ekegusii (guz_Latn) Datasets ===")
    
    # Check FLEURS / Bouquet / FineWeb / AfriSpeech / Masakhane
    dataset_configs = [
        ("facebook/bouquet", "guz_Latn"),
        ("google/fleurs", "guz_latn"),
        ("HuggingFaceFW/fineweb-2", "guz_Latn")
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    for ds_name, config in dataset_configs:
        url = f"https://datasets-server.huggingface.co/rows?dataset={ds_name}&config={config}&split=train&offset=0&limit=100"
        try:
            req = urllib.request.Request(url, headers=headers)
            res = urllib.request.urlopen(req).read().decode('utf-8')
            data = json.loads(res)
            rows = data.get('rows', [])
            print(f"Dataset '{ds_name}' ({config}): Found {len(rows)} rows sample!")
        except Exception as e:
            print(f"Dataset '{ds_name}' ({config}): {e}")

if __name__ == "__main__":
    search_and_download_hf_guz()
