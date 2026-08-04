"""
Fix Cell 3 in notebooks/00_setup_environment.ipynb
"""
import json, os

NOTEBOOK_PATH = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\notebooks\00_setup_environment.ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "Verifying dependencies" in src:
            cell["source"] = [
                "# ============================================================\n",
                "# CELL 3 — Verify all packages & auto-install missing ones\n",
                "# ============================================================\n",
                "print('=' * 60)\n",
                "print('Verifying dependencies & auto-installing missing packages...')\n",
                "print('=' * 60)\n",
                "\n",
                "import importlib, subprocess, sys, site, os, glob\n",
                "\n",
                "# Guarantees host conda site-packages & user site-packages are in sys.path\n",
                "user_site = site.getusersitepackages()\n",
                "if user_site not in sys.path:\n",
                "    sys.path.insert(0, user_site)\n",
                "\n",
                "for conda_site in glob.glob('/opt/conda/lib/python3.*/site-packages'):\n",
                "    if conda_site not in sys.path:\n",
                "        sys.path.insert(0, conda_site)\n",
                "\n",
                "REQUIRED = [\n",
                "    'torch', 'pandas', 'numpy', 'matplotlib',\n",
                "    'transformers', 'datasets', 'evaluate',\n",
                "    'peft', 'trl', 'bitsandbytes',\n",
                "    'sacrebleu', 'accelerate', 'tokenizers',\n",
                "    'rich', 'tqdm', 'scipy', 'sklearn', 'yaml',\n",
                "]\n",
                "\n",
                "PKG_MAP = {'yaml': 'pyyaml', 'sklearn': 'scikit-learn'}\n",
                "\n",
                "all_ok = True\n",
                "for mod in REQUIRED:\n",
                "    try:\n",
                "        m = importlib.import_module(mod)\n",
                "        ver = getattr(m, '__version__', 'installed')\n",
                "        print(f'  OK  {mod:<20} {ver}')\n",
                "    except ImportError:\n",
                "        pkg_name = PKG_MAP.get(mod, mod)\n",
                "        print(f'  Installing missing package: {pkg_name}...', end=' ', flush=True)\n",
                "        res = subprocess.run(\n",
                "            [sys.executable, '-m', 'pip', 'install', '--quiet', '--user', '--no-build-isolation', '--prefer-binary', pkg_name],\n",
                "            check=False,\n",
                "            capture_output=True,\n",
                "            text=True,\n",
                "        )\n",
                "        try:\n",
                "            importlib.invalidate_caches()\n",
                "            m = importlib.import_module(mod)\n",
                "            ver = getattr(m, '__version__', 'installed')\n",
                "            print(f'OK  ({ver})')\n",
                "        except ImportError as e:\n",
                "            print(f'FAILED: {e}')\n",
                "            all_ok = False\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "print(f'\\n  numpy  : {np.__version__}')\n",
                "print(f'  pandas : {pd.__version__}')\n",
                "print('\\n✅ All packages verified!' if all_ok else '\\n⚠️ Some packages missing!')\n"
            ]

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Updated 00_setup_environment.ipynb Cell 3!")
