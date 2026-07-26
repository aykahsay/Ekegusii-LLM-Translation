import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from datasets import Dataset

def main():
    model_checkpoint = "facebook/nllb-200-distilled-600M"
    output_dir = "models/nllb-en-guz"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading pretrained tokenizer & model from {model_checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, src_lang="eng_Latn", tgt_lang="guz_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    # 1. Load Ekegusii parallel dataset
    lang_file = os.path.join("data", "languages", "PSA_English_Ekegusii.csv")
    inter_train = os.path.join("data", "intermediate", "train_guz.csv")
    inter_dev = os.path.join("data", "intermediate", "dev_guz.csv")

    if os.path.exists(lang_file):
        print(f"Loading dataset from {lang_file}...")
        df = pd.read_csv(lang_file, dtype=str).dropna(subset=["English", "Ekegusii"])
        # 90% train / 10% dev split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(0.9 * len(df))
        train_df = df.iloc[:split_idx]
        dev_df = df.iloc[split_idx:]
    elif os.path.exists(inter_train) and os.path.exists(inter_dev):
        print(f"Loading fallback datasets from {inter_train} and {inter_dev}...")
        train_df = pd.read_csv(inter_train, dtype=str)
        dev_df = pd.read_csv(inter_dev, dtype=str)
    else:
        raise FileNotFoundError("No valid Ekegusii dataset found in data/languages/ or data/intermediate/")

    print(f"Train samples: {len(train_df)} | Dev samples: {len(dev_df)}")

    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    max_input_length = 128
    max_target_length = 128

    def preprocess_function(examples):
        inputs = [str(ex) for ex in examples["English"]]
        targets = [str(ex) for ex in examples["Ekegusii"]]
        
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_dev = dev_dataset.map(preprocess_function, batched=True, remove_columns=dev_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting NLLB-200 Ekegusii Fine-Tuning...")
    trainer.train()

    print(f"Saving fine-tuned Ekegusii model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
