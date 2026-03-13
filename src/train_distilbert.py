import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


#DATA_PATH = "data/processed/train_data_clean.csv"
DATA_PATH = "data/processed/train_data_bert_sample.csv"
MODEL_CHECKPOINT = "distilbert-base-uncased"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }


def main():
    # 1) Load data
    df = pd.read_csv(DATA_PATH)

    # Keep only needed columns
    df = df[["clean_text", "label", "source"]].dropna()
    df["label"] = df["label"].astype(int)
    df["clean_text"] = df["clean_text"].astype(str)

    # 2) Train/validation split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # 3) Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

    # 4) Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # 5) Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["clean_text"],
            truncation=True
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # 6) Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=2
    )

    # 7) Dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 8) Training config
    training_args = TrainingArguments(
        output_dir="distilbert-sentiment-output",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        report_to="none"
    )

    # 9) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        #tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 10) Train
    trainer.train()

    # 11) Final evaluation
    metrics = trainer.evaluate()
    print("\nFinal evaluation metrics:")
    print(metrics)


if __name__ == "__main__":
    main()