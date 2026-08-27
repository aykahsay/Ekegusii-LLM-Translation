import os
import sys

# Ensure UTF-8 output encoding for Windows terminal print statements
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN")

try:
    print("[1/5] Authenticating with Hugging Face Hub...")
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info['name']
    print(f"Authenticated as user: '{username}'")
    
    repo_id = f"{username}/Ekegusii-English-Kiswahili-Parallel-Corpus"
    print(f"Target repository ID: {repo_id}")
    
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)

    print("[2/5] Loading master corpus splits...")
    train_path = "data/master_corpus/splits/master_train.csv"
    val_path = "data/master_corpus/splits/master_val.csv"
    test_path = "data/master_corpus/splits/master_test.csv"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    dataset_dict = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    print(f"[3/5] Pushing DatasetDict splits (train, validation, test) to Hugging Face Hub...")
    dataset_dict.push_to_hub(repo_id, token=token)

    print("[4/5] Uploading raw CSV master corpus folder...")
    api.upload_folder(
        folder_path="data/master_corpus",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )

    print(f"\n[5/5] SUCCESS! Your dataset is now live at:")
    print(f"https://huggingface.co/datasets/{repo_id}")

except Exception as e:
    print(f"Error during Hugging Face upload: {e}")
    sys.exit(1)
