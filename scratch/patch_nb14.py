import json

path = 'notebooks/14_full_model_evaluation.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "# ============================================================\n",
    "# PATH & ENVIRONMENT BOOSTER — Guarantees project path setup\n",
    "# ============================================================\n",
    "import os, sys, site, glob\n",
    "\n",
    "try:\n",
    "    import typing_extensions\n",
    "    if not hasattr(typing_extensions, 'TypeIs'):\n",
    "        typing_extensions.TypeIs = getattr(typing_extensions, 'TypeGuard', getattr(typing_extensions, 'Any', object))\n",
    "except Exception:\n",
    "    pass\n",
    "\n",
    "user_site = site.getusersitepackages()\n",
    "if user_site not in sys.path:\n",
    "    sys.path.insert(0, user_site)\n",
    "for conda_site in glob.glob('/opt/conda/lib/python3.*/site-packages'):\n",
    "    if conda_site not in sys.path:\n",
    "        sys.path.append(conda_site)\n",
    "\n",
    "if os.path.basename(os.getcwd()) == 'notebooks':\n",
    "    os.chdir('..')\n",
    "\n",
    "if os.getcwd() not in sys.path:\n",
    "    sys.path.insert(0, os.getcwd())\n",
    "\n",
    "print(f'Working Directory : {os.getcwd()}')\n",
    "print(f'Python Kernel     : {sys.executable}')\n"
]

nb['cells'][1]['source'] = new_source
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully patched notebook 14!')
