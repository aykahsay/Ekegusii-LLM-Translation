"""
Update start.ipynb with robust module purge and site-packages setup
"""
import json, os

START_NB = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\start.ipynb"

with open(START_NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    src = "".join(cell.get("source", []))
    if "FRESH CLONE" in src or "urllib.request.urlretrieve" in src:
        cell["source"] = [
            "# ============================================================\n",
            "# FRESH CLONE & ENVIRONMENT SETUP\n",
            "# ============================================================\n",
            "import urllib.request, zipfile, os, sys, site, glob\n",
            "\n",
            "# Step 1: Purge any cached 'src' modules from memory\n",
            "for mod in list(sys.modules.keys()):\n",
            "    if mod.startswith('src'):\n",
            "        del sys.modules[mod]\n",
            "\n",
            "# Step 2: Go to home directory\n",
            "home = os.path.expanduser('~')\n",
            "os.chdir(home)\n",
            "print(f'📍 Starting from home: {os.getcwd()}')\n",
            "\n",
            "# Step 3: Download fresh repo ZIP from GitHub\n",
            "print('📥 Downloading latest repository code from GitHub main branch...')\n",
            "zip_path = os.path.join(home, 'repo.zip')\n",
            "urllib.request.urlretrieve(\n",
            "    'https://github.com/aykahsay/Ekegusii-LLM-Translation/archive/refs/heads/main.zip',\n",
            "    zip_path\n",
            ")\n",
            "print('✅ Download complete!')\n",
            "\n",
            "# Step 4: Extract and enter project directory\n",
            "print('📦 Extracting...')\n",
            "with zipfile.ZipFile(zip_path, 'r') as z:\n",
            "    z.extractall(home)\n",
            "os.remove(zip_path)\n",
            "print('✅ Extracted!')\n",
            "\n",
            "proj_dir = os.path.join(home, 'Ekegusii-LLM-Translation-main')\n",
            "os.chdir(proj_dir)\n",
            "if proj_dir not in sys.path:\n",
            "    sys.path.insert(0, proj_dir)\n",
            "\n",
            "# Step 5: Configure sys.path for user and conda site-packages\n",
            "user_site = site.getusersitepackages()\n",
            "if user_site not in sys.path:\n",
            "    sys.path.insert(0, user_site)\n",
            "for conda_site in glob.glob('/opt/conda/lib/python3.*/site-packages'):\n",
            "    if conda_site not in sys.path:\n",
            "        sys.path.insert(0, conda_site)\n",
            "\n",
            "print(f'\\n📁 Working Directory : {os.getcwd()}')\n",
            "print(f'🐍 Python Executable  : {sys.executable}')\n",
            "print('\\n✅ FRESH CLONE & SETUP COMPLETE!')\n",
            "print('🚀 You can now open any notebook in notebooks/ (e.g. notebooks/00_setup_environment.ipynb)!')\n"
        ]

with open(START_NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Updated start.ipynb!")
