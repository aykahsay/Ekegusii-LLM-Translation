import urllib.request
import json

def search_hf_guz():
    # HuggingFace Datasets API for guz_Latn
    urls = [
        "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/finetranslations&config=guz_Latn&split=train&offset=0&limit=10",
        "https://datasets-server.huggingface.co/splits?dataset=HuggingFaceFW/finetranslations",
        "https://datasets-server.huggingface.co/splits?dataset=facebook/flores"
    ]
    headers = {'User-Agent': 'Mozilla/5.0'}
    for u in urls:
        try:
            req = urllib.request.Request(u, headers=headers)
            res = urllib.request.urlopen(req).read().decode('utf-8')
            data = json.loads(res)
            print(f"URL: {u[:80]}... SUCCESS!")
            if 'splits' in data:
                guz_splits = [s for s in data['splits'] if 'guz' in s.get('config', '') or 'guz' in s.get('split', '')]
                print(f"Found {len(guz_splits)} guz splits: {guz_splits[:5]}")
            if 'rows' in data:
                print(f"Rows count: {len(data['rows'])}")
                print(data['rows'][0]['row'])
        except Exception as e:
            print(f"URL: {u[:80]}... Error: {e}")

if __name__ == "__main__":
    search_hf_guz()
