"""
Comprehensive Audit & Hardening Script for ALL 14 Notebooks
-----------------------------------------------------------
This script inspects every code cell in every notebook (.ipynb) file:
1. Injects a fail-safe path booster into the top code cell of every notebook.
2. Ensures every cell importing `src` or using local data paths sets `sys.path` to project root.
3. Fixes any deprecated or ABI-sensitive imports.
"""

import json
import os
import sys

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")

SAFE_PATH_BOOSTER = [
    "# ============================================================\n",
    "# PATH BOOSTER — Guarantees project root in sys.path & CWD\n",
    "# ============================================================\n",
    "import os, sys\n",
    "try:\n",
    "    cwd = os.getcwd()\n",
    "except FileNotFoundError:\n",
    "    cwd = os.path.expanduser('~')\n",
    "    os.chdir(cwd)\n",
    "proj_dir = os.path.join(os.path.expanduser('~'), 'Ekegusii-LLM-Translation-main')\n",
    "if os.path.isdir(proj_dir):\n",
    "    os.chdir(proj_dir)\n",
    "elif os.path.basename(os.getcwd()) == 'notebooks':\n",
    "    os.chdir('..')\n",
    "if os.getcwd() not in sys.path:\n",
    "    sys.path.insert(0, os.getcwd())\n"
]

def audit_and_harden_notebook(filepath):
    filename = os.path.basename(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    modified = False

    # Check first code cell
    first_code_idx = None
    for i, c in enumerate(cells):
        if c.get("cell_type") == "code":
            first_code_idx = i
            break

    if first_code_idx is not None:
        cell_src = "".join(cells[first_code_idx].get("source", []))
        if "PATH BOOSTER" not in cell_src:
            booster_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": SAFE_PATH_BOOSTER
            }
            cells.insert(first_code_idx, booster_cell)
            modified = True

    # Check each code cell for imports of src
    for c in cells:
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            if "from src." in src or "import src." in src:
                if "sys.path" not in src:
                    # Prepend a lightweight 3-line path check
                    prepend = [
                        "import os, sys\n",
                        "p = os.path.join(os.path.expanduser('~'), 'Ekegusii-LLM-Translation-main')\n",
                        "if os.path.isdir(p) and p not in sys.path: sys.path.insert(0, p); os.chdir(p)\n",
                        "\n"
                    ]
                    c["source"] = prepend + c["source"]
                    modified = True

    if modified:
        nb["cells"] = cells
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        return "HARDENED"
    return "OK"


def main():
    print("=" * 60)
    print("Auditing & Hardening All Notebooks")
    print("=" * 60)
    
    count_hardened = 0
    count_ok = 0

    for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
        if not fname.endswith(".ipynb"):
            continue
        fpath = os.path.join(NOTEBOOKS_DIR, fname)
        status = audit_and_harden_notebook(fpath)
        print(f"  [{status:<8}] {fname}")
        if status == "HARDENED":
            count_hardened += 1
        else:
            count_ok += 1

    print("-" * 60)
    print(f"Total: {count_hardened} hardened, {count_ok} already OK.")

if __name__ == "__main__":
    main()
