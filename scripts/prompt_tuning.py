# scripts/prompt_tuning.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import os

MODEL_NAME = "gpt2"
DATA_PATH = "data/custom_data.csv"
OUTPUT_DIR = "./models"


def tokenize_function(example, tokenizer):
    tokens = tokenizer(example['text'], padding="max_length", truncation=True, max_length=64)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} does not exist.")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset('csv', data_files={'train': DATA_PATH})
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, overwrite_output_dir=True, num_train_epochs=5,
        per_device_train_batch_size=1, save_steps=100, save_total_limit=2, prediction_loss_only=False
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets['train'])
    print("Starting prompt tuning with custom data...")
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Model saved to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
