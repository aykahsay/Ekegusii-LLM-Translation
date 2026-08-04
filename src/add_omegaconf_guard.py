"""
Update Notebooks Package Verification & Auto-Install
----------------------------------------------------
Ensures `omegaconf` is included in all notebook package checks, and
adds auto-install for `omegaconf==2.3.0` + `hydra-core==1.3.2` if missing.
"""

import json
import os
import sys

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")

AUTO_INSTALL_OMEGACONF_CODE = (
    "import sys, subprocess, importlib\n"
    "for mod, pkg in [('omegaconf', 'omegaconf==2.3.0'), ('hydra', 'hydra-core==1.3.2')]:\n"
    "    try:\n"
    "        importlib.import_module(mod)\n"
    "    except ImportError:\n"
    "        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', pkg], check=False)\n"
)

def update_notebook(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    modified = False

    for cell in cells:
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            # Inject omegaconf check into ABI / import check cells
            if "REQUIRED = [" in src and "'omegaconf'" not in src:
                src = src.replace(
                    "'sklearn',",
                    "'sklearn', 'omegaconf',"
                )
                cell["source"] = [src]
                modified = True

            # If cell imports omegaconf or config.py, ensure omegaconf auto-installer runs
            if ("from omegaconf" in src or "from src.utils.config" in src or "from src.tokenizer" in src) and "importlib.import_module('omegaconf')" not in src:
                if "import sys" not in src:
                    cell["source"] = [AUTO_INSTALL_OMEGACONF_CODE + "\n"] + cell["source"]
                    modified = True

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        return True
    return False


def main():
    print("=== Updating Notebooks for omegaconf auto-installation ===")
    updated = 0
    for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
        if fname.endswith(".ipynb"):
            fpath = os.path.join(NOTEBOOKS_DIR, fname)
            if update_notebook(fpath):
                print(f"  [UPDATED] {fname}")
                updated += 1
            else:
                print(f"  [OK]      {fname}")
    print(f"\nDone: {updated} notebooks updated.")

if __name__ == "__main__":
    main()
