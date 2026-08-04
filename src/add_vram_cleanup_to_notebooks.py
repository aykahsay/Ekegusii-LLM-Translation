"""
Add GPU VRAM cleanup to 07_train_qwen.ipynb and 08_train_mistral.ipynb
"""
import json, os

NOTEBOOKS_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\notebooks"

for fname in ["07_train_qwen.ipynb", "08_train_mistral.ipynb"]:
    path = os.path.join(NOTEBOOKS_DIR, fname)
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "run_train(" in src and "gc.collect()" not in src:
                lines = cell["source"]
                if isinstance(lines, str):
                    lines = [lines]
                new_src = [
                    "import gc, torch\n",
                    "gc.collect()\n",
                    "if torch.cuda.is_available():\n",
                    "    torch.cuda.empty_cache()\n",
                    "\n"
                ] + lines
                cell["source"] = new_src

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Updated VRAM cleanup in {fname}")
