"""
Fix All Research Notebook Path Boosters
-----------------------------------------
Replaces the old fragile path booster (which breaks on Kineses when
os.getcwd() raises FileNotFoundError) with a robust home-based booster
that works on ANY Jupyter environment including Kineses Cloud.
"""

import os
import json

WORKSPACE_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp"
NOTEBOOKS_DIR = os.path.join(WORKSPACE_DIR, "notebooks")

# Old booster signature to detect & replace
OLD_BOOSTER_MARKER = "AUTOMATIC WORKING DIRECTORY & IMPORT PATH BOOSTER"

# New robust booster that works on Kineses (handles invalid CWD gracefully)
NEW_PATH_BOOSTER_CELL = {
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
        "# Step 1: Safely get current directory (may throw on Kineses)\n",
        "try:\n",
        "    cwd = os.getcwd()\n",
        "except FileNotFoundError:\n",
        "    cwd = os.path.expanduser('~')\n",
        "    os.chdir(cwd)\n",
        "\n",
        "# Step 2: Always try the known Kineses project path first\n",
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
        "print(f'✅ Working Directory : {os.getcwd()}')\n",
        "print(f'✅ Python Kernel     : {sys.executable}')"
    ]
}


def fix_all_notebooks():
    print("=== Fixing Path Boosters in All Research Notebooks ===\n")

    fixed = 0
    skipped = 0

    for filename in sorted(os.listdir(NOTEBOOKS_DIR)):
        if not filename.endswith(".ipynb"):
            continue

        filepath = os.path.join(NOTEBOOKS_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb.get("cells", [])
        new_cells = []
        replaced = False
        skip_next = False

        for i, cell in enumerate(cells):
            src = "".join(cell.get("source", []))

            # Remove old broken booster code cell
            if cell.get("cell_type") == "code" and OLD_BOOSTER_MARKER in src:
                new_cells.append(NEW_PATH_BOOSTER_CELL)
                replaced = True
                print(f"  [REPLACED] {filename}")
                continue

            new_cells.append(cell)

        # If no old booster found, inject after title cell
        if not replaced:
            # Check if new booster already present
            has_new = any(
                "PATH BOOSTER" in "".join(c.get("source", []))
                for c in cells
                if c.get("cell_type") == "code"
            )
            if not has_new:
                new_cells.insert(1, NEW_PATH_BOOSTER_CELL)
                print(f"  [INJECTED] {filename}")
                replaced = True
            else:
                print(f"  [OK]       {filename} — already has new booster")
                skipped += 1

        nb["cells"] = new_cells

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)

        if replaced:
            fixed += 1

    print(f"\n=== Done: {fixed} notebooks fixed, {skipped} already OK ===")


if __name__ == "__main__":
    fix_all_notebooks()
