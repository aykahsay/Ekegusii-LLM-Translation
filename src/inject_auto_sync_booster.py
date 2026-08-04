"""
Permanent Auto-Sync Notebook Guard Script
-------------------------------------------
Adds an auto-sync check to Cell 1 of all research notebooks.
If `configs/models/v2_mistral_earlystop_v5.tag` is missing (outdated clone on Kineses),
it automatically downloads the latest zip from GitHub, extracts it, and purges sys.modules.
"""

import json
import os
import sys

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")

TAG_FILE = os.path.join(WORKSPACE_DIR, "configs", "models", "v2_mistral_earlystop_v5.tag")
with open(TAG_FILE, "w", encoding="utf-8") as f:
    f.write("v5")

AUTO_SYNC_BOOSTER = [
    "# ============================================================\n",
    "# PATH & REPO AUTO-SYNC BOOSTER — Guarantees latest project code\n",
    "# ============================================================\n",
    "import os, sys, site, urllib.request, zipfile\n",
    "\n",
    "# Purge cached 'src' modules from memory so updated files take effect immediately\n",
    "for mod in list(sys.modules.keys()):\n",
    "    if mod.startswith('src'):\n",
    "        del sys.modules[mod]\n",
    "\n",
    "user_site = site.getusersitepackages()\n",
    "if user_site not in sys.path:\n",
    "    sys.path.insert(0, user_site)\n",
    "\n",
    "try:\n",
    "    cwd = os.getcwd()\n",
    "except FileNotFoundError:\n",
    "    cwd = os.path.expanduser('~')\n",
    "    os.chdir(cwd)\n",
    "\n",
    "home       = os.path.expanduser('~')\n",
    "proj_dir   = os.path.join(home, 'Ekegusii-LLM-Translation-main')\n",
    "sync_tag   = os.path.join(proj_dir, 'configs', 'models', 'v2_mistral_earlystop_v5.tag')\n",
    "\n",
    "# Auto-sync if folder is missing OR outdated (lacks v2_mistral_earlystop_v5.tag)\n",
    "if not os.path.isfile(sync_tag):\n",
    "    print('🔄 Outdated or missing repository detected. Auto-syncing latest code from GitHub...')\n",
    "    zip_path = os.path.join(home, 'repo.zip')\n",
    "    urllib.request.urlretrieve('https://github.com/aykahsay/Ekegusii-LLM-Translation/archive/refs/heads/main.zip', zip_path)\n",
    "    with zipfile.ZipFile(zip_path, 'r') as z:\n",
    "        z.extractall(home)\n",
    "    os.remove(zip_path)\n",
    "    print('✅ Repository auto-synced to latest main commit!')\n",
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
    print("=== Injecting v5 Auto-Sync Booster into Notebooks ===")
    for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
        if fname.endswith(".ipynb"):
            fpath = os.path.join(NOTEBOOKS_DIR, fname)
            update_notebook_booster(fpath)
            print(f"  [v5 AUTO-SYNC ADDED] {fname}")

if __name__ == "__main__":
    main()
