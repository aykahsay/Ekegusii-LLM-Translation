"""
Permanent Auto-Sync Notebook Guard Script
-------------------------------------------
Adds an auto-sync check to Cell 1 of all research notebooks.
Only downloads if `configs/models/v5_ready.tag` is missing, preventing
file modifications during active training runs.
"""

import json
import os
import sys

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")

TAG_FILE = os.path.join(WORKSPACE_DIR, "configs", "models", "v5_ready.tag")
with open(TAG_FILE, "w", encoding="utf-8") as f:
    f.write("v5_ready")

AUTO_SYNC_BOOSTER = [
    "# ============================================================\n",
    "# PATH & ENVIRONMENT BOOSTER — Guarantees project path setup\n",
    "# ============================================================\n",
    "import os, sys, site, urllib.request, zipfile, glob\n",
    "\n",
    "user_site = site.getusersitepackages()\n",
    "if user_site not in sys.path:\n",
    "    sys.path.insert(0, user_site)\n",
    "for conda_site in glob.glob('/opt/conda/lib/python3.*/site-packages'):\n",
    "    if conda_site not in sys.path:\n",
    "        sys.path.insert(0, conda_site)\n",
    "\n",
    "try:\n",
    "    cwd = os.getcwd()\n",
    "except FileNotFoundError:\n",
    "    cwd = os.path.expanduser('~')\n",
    "    os.chdir(cwd)\n",
    "\n",
    "home     = os.path.expanduser('~')\n",
    "proj_dir = os.path.join(home, 'Ekegusii-LLM-Translation-main')\n",
    "tag_file = os.path.join(proj_dir, 'configs', 'models', 'v5_ready.tag')\n",
    "\n",
    "# Download ONLY if tag_file is missing (prevents file modification during active runs)\n",
    "if not os.path.isfile(tag_file):\n",
    "    try:\n",
    "        print('🔄 Syncing code from GitHub main branch...')\n",
    "        zip_path = os.path.join(home, 'repo.zip')\n",
    "        urllib.request.urlretrieve('https://github.com/aykahsay/Ekegusii-LLM-Translation/archive/refs/heads/main.zip', zip_path)\n",
    "        with zipfile.ZipFile(zip_path, 'r') as z:\n",
    "            z.extractall(home)\n",
    "        os.remove(zip_path)\n",
    "        print('✅ Code synced to latest version!')\n",
    "    except Exception as exc:\n",
    "        print(f'⚠️ Notice: {exc} (using local files)')\n",
    "else:\n",
    "    print('✅ Codebase up to date.')\n",
    "\n",
    "if os.path.isdir(proj_dir):\n",
    "    os.chdir(proj_dir)\n",
    "elif os.path.basename(os.getcwd()) == 'notebooks':\n",
    "    os.chdir('..')\n",
    "\n",
    "if os.getcwd() not in sys.path:\n",
    "    sys.path.insert(0, os.getcwd())\n",
    "\n",
    "print(f'Working Directory : {os.getcwd()}')\n",
    "print(f'Python Kernel     : {sys.executable}')\n"
]


def update_notebook_booster(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    first_code_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "cell_type":
            pass
        if c.get("cell_type") == "code":
            first_code_idx = i
            break

    if first_code_idx is not None:
        cells[first_code_idx] = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": AUTO_SYNC_BOOSTER
        }

    nb["cells"] = cells
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)


def main():
    print("=== Injecting Non-Disruptive Booster into Notebooks ===")
    for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
        if fname.endswith(".ipynb"):
            fpath = os.path.join(NOTEBOOKS_DIR, fname)
            update_notebook_booster(fpath)
            print(f"  [BOOSTER UPDATED] {fname}")

if __name__ == "__main__":
    main()
