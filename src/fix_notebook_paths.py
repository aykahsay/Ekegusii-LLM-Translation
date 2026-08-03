"""
Automatic Notebook Working Directory Fixer
------------------------------------------
Injects automatic root path detection into all 13 Jupyter notebooks so that:
1. Relative paths to data/master_corpus/ work seamlessly whether running from root or notebooks/.
2. import src... works seamlessly without needing pip install -e . or git installed.
"""

import os
import json

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")

PATH_FIX_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# AUTOMATIC WORKING DIRECTORY & IMPORT PATH BOOSTER\n",
        "# Works seamlessly in Jupyter Notebook / Lab / Google Colab / Kaggle without git\n",
        "import os\n",
        "import sys\n\n",
        "if os.path.basename(os.getcwd()) == 'notebooks':\n",
        "    os.chdir('..')\n",
        "if os.getcwd() not in sys.path:\n",
        "    sys.path.append(os.getcwd())\n\n",
        "print('Current Working Directory:', os.getcwd())\n",
        "print('Python Path Configured Successfully!')"
    ]
}

def fix_all_notebooks():
    print("=== Injecting Automatic Root Path Fix into All 13 Notebooks ===")
    
    for filename in sorted(os.listdir(NOTEBOOKS_DIR)):
        if filename.endswith(".ipynb"):
            filepath = os.path.join(NOTEBOOKS_DIR, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                nb = json.load(f)
                
            # Check if path fix is already injected
            has_fix = False
            for cell in nb.get("cells", []):
                if cell.get("cell_type") == "code" and "AUTOMATIC WORKING DIRECTORY" in "".join(cell.get("source", [])):
                    has_fix = True
                    break
                    
            if not has_fix:
                # Insert path fix cell right after markdown title cell
                nb["cells"].insert(1, PATH_FIX_CELL)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(nb, f, indent=2)
                    
                print(f"[OK] Injected path booster into: {filename}")
            else:
                print(f"[SKIP] Path booster already present in: {filename}")

if __name__ == "__main__":
    fix_all_notebooks()
