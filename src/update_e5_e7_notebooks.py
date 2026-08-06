"""
Update notebooks/07_train_qwen.ipynb and 08_train_mistral.ipynb cell 5 to include the explicit E5-E7 loop.
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
        if "TRAINABLE_EXPERIMENTS" in src or "EXPERIMENTS_5_TO_7" in src:
            cell["source"] = [
                "import gc, torch\n",
                "from src.cli.train import run_train\n",
                "\n",
                "# Experiments E5, E6, E7 list\n",
                "EXPERIMENTS_5_TO_7 = [\n",
                "    'E5_Full_Resources',          # 50% Data Scaling Ablation\n",
                "    'E6_Lexical_Augmentation',    # Translation-Only Ablation (No Dictionary)\n",
                "    'E7_Curriculum_Learning',     # Dictionary-Heavy Ablation\n",
                "]\n",
                "\n",
                "for exp_id in EXPERIMENTS_5_TO_7:\n",
                "    print('\\n' + '=' * 65)\n",
                "    print(f'🚀 STARTING EXPERIMENT: {exp_id}')\n",
                "    print('=' * 65)\n",
                "    gc.collect()\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.empty_cache()\n",
                "    run_train(exp_id, model_name='" + model_key + "')\n",
                "    gc.collect()\n",
                "    if torch.cuda.is_available():\n",
                "        torch.cuda.empty_cache()\n",
                "\n",
                "print('\\n🎉 ALL ABLATION EXPERIMENTS E5, E6, AND E7 ARE COMPLETE!')\n"
            ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Updated {fname} with E5-E7 runner code")
