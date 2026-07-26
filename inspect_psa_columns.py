"""
inspect_psa_columns.py
----------------------
Inspects every file in data/ to detail its columns, row count,
and text fields for combining.
"""
import sys, io, os, json
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DATA_DIR = "data"

def inspect():
    print("=" * 80)
    print("  DETAILED COLUMN AND STRUCTURE ANALYSIS OF ALL PSA DATASETS")
    print("=" * 80)
    
    files = sorted(os.listdir(DATA_DIR))
    
    summary = []
    
    for fname in files:
        fpath = os.path.join(DATA_DIR, fname)
        if fname.endswith(".csv"):
            try:
                df = pd.read_csv(fpath, on_bad_lines="skip", dtype=str)
                cols = list(df.columns)
                sample = df.iloc[0].to_dict() if len(df) > 0 else {}
                summary.append({
                    "File": fname,
                    "Type": "CSV",
                    "Rows": len(df),
                    "Columns_Count": len(cols),
                    "Columns": cols,
                    "Sample": sample
                })
            except Exception as e:
                print(f"[!] Error reading {fname}: {e}")
        elif fname.endswith(".json"):
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    sample = data[0] if data else {}
                    cols = list(sample.keys()) if isinstance(sample, dict) else ["item"]
                    summary.append({
                        "File": fname,
                        "Type": "JSON (List)",
                        "Rows": len(data),
                        "Columns_Count": len(cols),
                        "Columns": cols,
                        "Sample": sample
                    })
                elif isinstance(data, dict):
                    keys = list(data.keys())
                    first_val = data[keys[0]] if keys else {}
                    summary.append({
                        "File": fname,
                        "Type": "JSON (Dict)",
                        "Rows": len(keys),
                        "Columns_Count": 1,
                        "Columns": ["dict_key -> " + type(first_val).__name__],
                        "Sample": {keys[0]: str(first_val)[:100]}
                    })
            except Exception as e:
                print(f"[!] Error reading {fname}: {e}")

    for item in summary:
        print(f"\n📄  {item['File']} ({item['Type']})")
        print(f"    Total Rows : {item['Rows']}")
        print(f"    Columns ({item['Columns_Count']}) : {item['Columns']}")
        print("    Sample Row:")
        for k, v in list(item['Sample'].items())[:4]:
            val_str = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
            print(f"      • {k}: {val_str}")

if __name__ == "__main__":
    inspect()
