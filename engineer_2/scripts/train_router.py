import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

def prepare_dataset(data_path: str, tokenizer):
    with open(data_path, "r") as f:
        raw_data = json.load(f)

    formatted_data = {
        "input_text": [
            f"Classify retrieval requirement (0: none, 1: single-step, 2: multi-step): {item['query']}"
            for item in raw_data
        ],
        "target_text": [str(item["label"]) for item in raw_data]
    }
    
    dataset = Dataset.from_dict(formatted_data)
    
    def preprocess_function(examples):
        model_inputs = tokenizer(examples["input_text"], max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(examples["target_text"], max_length=4, truncation=True, padding="max_length")
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)

def train():
    model_id = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    dataset = prepare_dataset("engineer_2/configs/sample_labeled_queries.json", tokenizer)
    split_dataset = dataset.train_test_split(test_size=0.1)

    training_args = Seq2SeqTrainingArguments(
        output_dir="./router_flan_t5_checkpoint",
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    trainer.train()
    trainer.save_model("./final_router_checkpoint")
    tokenizer.save_pretrained("./final_router_checkpoint")
    print("Checkpoint saved to ./final_router_checkpoint")

if __name__ == "__main__":
    train()
