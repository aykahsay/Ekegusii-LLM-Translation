"""
Update notebooks/07_train_qwen.ipynb and 08_train_mistral.ipynb to automatically skip E4_Trilingual in the all-experiments loop.
"""
import json, os

NOTEBOOKS_DIR = r"c:\Users\Admin\OneDrive - United States International University (USIU)\Documents\NLP\Multilogual_transaltion_nlp\notebooks"

for fname in ["07_train_qwen.ipynb", "08_train_mistral.ipynb"]:
    path = os.path.join(NOTEBOOKS_DIR, fname)
    model_key = "qwen" if "qwen" in fname else "mistral"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "TRAINABLE_EXPERIMENTS" in src and "for experiment_id in" in src:
            cell["source"] = [
                "from src.cli.train import TRAINABLE_EXPERIMENTS, run_train\n",
                "import gc, torch\n",
                "\n",
                "for experiment_id in TRAINABLE_EXPERIMENTS:\n",
                "    if experiment_id == 'E4_Trilingual':\n",
                "        print('⏩ Skipping E4_Trilingual (Already completed!)')\n",
                "        continue\n",
                "    print(f'=== Training " + model_key + " on {experiment_id} ===')\n",
                "    run_train(experiment_id, model_name='" + model_key + "')\n",
                "    gc.collect()\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.empty_cache()\n"
            ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Updated loop in {fname} to skip E4_Trilingual")
