import urllib.request
import json
import re
import os
import pandas as pd

def clean_text(text):
    if not text:
        return ""
    # remove single quotes wrapping individual letters or quotes
    text = re.sub(r"'([^']*)'", r"\1", text)
    text = re.sub(r"'+", "", text)
    text = re.sub(r'"+', "", text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_markdown_sections(content):
    # split by ## section headers
    sections = re.split(r'##\s*', content)
    cleaned_sections = []
    for sec in sections:
        lines = sec.strip().split('\n')
        # skip title line if header #
        text_lines = [l.strip() for l in lines if not l.strip().startswith('#') and l.strip()]
        full_text = clean_text(" ".join(text_lines))
        if full_text and len(full_text) > 3:
            cleaned_sections.append(full_text)
    return cleaned_sections

def download_and_process_storybooks():
    print("=== Fetching African Storybook Parallel Corpus (EN - GUZ - SW) ===")
    tree_url = "https://api.github.com/repos/global-asp/asp-source/git/trees/master?recursive=1"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    req = urllib.request.Request(tree_url, headers=headers)
    res = urllib.request.urlopen(req).read().decode('utf-8')
    data = json.loads(res)
    
    tree = data.get('tree', [])
    guz_files = [item['path'] for item in tree if item['path'].startswith('guz/')]
    en_files = {item['path'].split('/')[-1].split('_')[0]: item['path'] for item in tree if item['path'].startswith('en/')}
    sw_files = {item['path'].split('/')[-1].split('_')[0]: item['path'] for item in tree if item['path'].startswith('sw/')}
    
    base_raw_url = "https://raw.githubusercontent.com/global-asp/asp-source/master/"
    
    all_trilingual_rows = []
    
    for guz_path in guz_files:
        filename = guz_path.split('/')[-1]
        story_id = filename.split('_')[0]
        
        en_path = en_files.get(story_id)
        sw_path = sw_files.get(story_id)
        
        if not en_path or not sw_path:
            continue
            
        print(f"Processing Story ID {story_id}: {filename}...")
        
        # Download files
        guz_content = urllib.request.urlopen(urllib.request.Request(base_raw_url + guz_path, headers=headers)).read().decode('utf-8')
        en_content = urllib.request.urlopen(urllib.request.Request(base_raw_url + en_path, headers=headers)).read().decode('utf-8')
        sw_content = urllib.request.urlopen(urllib.request.Request(base_raw_url + sw_path, headers=headers)).read().decode('utf-8')
        
        guz_secs = parse_markdown_sections(guz_content)
        en_secs = parse_markdown_sections(en_content)
        sw_secs = parse_markdown_sections(sw_content)
        
        min_len = min(len(guz_secs), len(en_secs), len(sw_secs))
        for i in range(min_len):
            all_trilingual_rows.append({
                'english': en_secs[i],
                'ekegusii': guz_secs[i],
                'swahili': sw_secs[i]
            })
            
    df = pd.DataFrame(all_trilingual_rows)
    print(f"\nTotal parallel sections aligned: {len(df)}")
    
    out_dir = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\data\clean"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save Trilingual
    tri_path = os.path.join(out_dir, "African_Storybooks_Trilingual.csv")
    df[['english', 'ekegusii', 'swahili']].to_csv(tri_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {tri_path}")
    
    # Save English-Ekegusii
    eng_guz_path = os.path.join(out_dir, "African_Storybooks_English_Ekegusii.csv")
    df[['english', 'ekegusii']].to_csv(eng_guz_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {eng_guz_path}")

    # Save Ekegusii-Swahili
    guz_sw_path = os.path.join(out_dir, "African_Storybooks_Ekegusii_Swahili.csv")
    df[['ekegusii', 'swahili']].to_csv(guz_sw_path, index=False, encoding='utf-8-sig')
    print(f"Saved: {guz_sw_path}")

if __name__ == "__main__":
    download_and_process_storybooks()
