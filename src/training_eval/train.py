import os
import pandas as pd
from datasets import Dataset
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

def load_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, dtype=str)
    return Dataset.from_pandas(df)

def preprocess_function(examples, tokenizer, max_length=128):
    inputs = [str(ex) for ex in examples["English"]]
    targets = [str(ex) for ex in examples["Kiswahili"]]
    
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, padding="max_length", truncation=True)
    
    model_inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
    ]
    return model_inputs

def main():
    model_checkpoint = "Helsinki-NLP/opus-mt-en-sw"
    output_dir = "models/psa-en-sw-finetuned"
    
    print(f"Loading tokenizer and model: {model_checkpoint}")
    tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    
    lang_file = os.path.join("data", "languages", "PSA_English_Swahili.csv")
    inter_train = os.path.join("data", "intermediate", "train.csv")
    inter_dev = os.path.join("data", "intermediate", "dev.csv")

    if os.path.exists(lang_file):
        print(f"Loading datasets from {lang_file}...")
        df = pd.read_csv(lang_file, dtype=str).dropna(subset=["English", "Kiswahili"])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(0.9 * len(df))
        train_dataset = Dataset.from_pandas(df.iloc[:split_idx])
        dev_dataset = Dataset.from_pandas(df.iloc[split_idx:])
    elif os.path.exists(inter_train) and os.path.exists(inter_dev):
        print(f"Loading datasets from {inter_train} and {inter_dev}...")
        train_dataset = load_data(inter_train)
        dev_dataset = load_data(inter_dev)
    else:
        raise FileNotFoundError("No valid Swahili dataset found in data/languages/ or data/intermediate/")
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    tokenized_dev = dev_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=dev_dataset.column_names)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=False,
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
    
    print("\nStarting Swahili Opus-MT Fine-Tuning...")
    trainer.train()
    
    print(f"Saving fine-tuned model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
