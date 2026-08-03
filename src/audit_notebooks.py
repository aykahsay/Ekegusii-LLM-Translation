"""
Full notebook directory audit and repair script.
Ensures all 13 research notebooks have the correct robust Kineses path booster.
"""
import json
import os
import sys

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")
PROJ_DIR = "Ekegusii-LLM-Translation-main"

BOOSTER_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# PATH BOOSTER — Works on Kineses / Jupyter / Colab / Kaggle\n",
        "# Handles FileNotFoundError when kernel CWD no longer exists.\n",
        "# ============================================================\n",
        "import os, sys\n",
        "\n",
        "# Step 1: Safely get current directory\n",
        "try:\n",
        "    cwd = os.getcwd()\n",
        "except FileNotFoundError:\n",
        "    cwd = os.path.expanduser('~')\n",
        "    os.chdir(cwd)\n",
        "\n",
        "# Step 2: Navigate to project root (Kineses home-based path)\n",
        "kineses_proj = os.path.join(os.path.expanduser('~'), 'Ekegusii-LLM-Translation-main')\n",
        "if os.path.isdir(kineses_proj):\n",
        "    os.chdir(kineses_proj)\n",
        "elif os.path.basename(os.getcwd()) == 'notebooks':\n",
        "    os.chdir('..')\n",
        "\n",
        "# Step 3: Add project root to Python path\n",
        "proj_root = os.getcwd()\n",
        "if proj_root not in sys.path:\n",
        "    sys.path.insert(0, proj_root)\n",
        "\n",
        "print(f'Working Directory : {os.getcwd()}')\n",
        "print(f'Python Kernel     : {sys.executable}')\n"
    ]
}

BOOSTER_MARKER = "PATH BOOSTER"


def has_booster(nb):
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if BOOSTER_MARKER in src and "kineses_proj" in src:
            return True
    return False


def remove_old_booster(cells):
    """Remove any old broken booster cells."""
    cleaned = []
    for cell in cells:
        src = "".join(cell.get("source", []))
        is_old_booster = (
            cell.get("cell_type") == "code"
            and (
                "AUTOMATIC WORKING DIRECTORY" in src
                or ("PATH BOOSTER" in src and "kineses_proj" not in src)
            )
        )
        if not is_old_booster:
            cleaned.append(cell)
    return cleaned


def fix_notebook(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    cells = remove_old_booster(cells)

    if not has_booster({"cells": cells}):
        # Insert after title markdown cell
        insert_pos = 1
        for i, cell in enumerate(cells):
            if cell.get("cell_type") == "markdown":
                insert_pos = i + 1
                break
        cells.insert(insert_pos, BOOSTER_CELL)
        nb["cells"] = cells
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2, ensure_ascii=False)
        return "FIXED"
    else:
        return "OK"


def main():
    print("=" * 60)
    print("Notebook Directory Audit & Repair")
    print("=" * 60)

    for fname in sorted(os.listdir(NOTEBOOKS_DIR)):
        if not fname.endswith(".ipynb"):
            continue
        if fname == "00_setup_environment.ipynb":
            continue  # setup nb is handled separately

        fpath = os.path.join(NOTEBOOKS_DIR, fname)
        result = fix_notebook(fpath)
        print(f"  [{result}] {fname}")

    print("\nAll notebooks processed.")


if __name__ == "__main__":
    main()
