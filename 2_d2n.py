import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Define the paths
dataset_path = "/home/gpu/Medical_app/T5_dh/medical/encoded_data/d2n"

# Load the tokenizer and model
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Load the dataset
def preprocess_function(examples, include_output=False):
    inputs = examples["inputs"]
    targets = examples["target"]

    # Tokenize inputs and targets
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        targets, max_length=512, truncation=True, padding="max_length"
    ).input_ids

    # Replace padding token ID with -100 for labels
    labels = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels
    ]
    model_inputs["labels"] = labels

    # Handle additional output if present (only for training)
    if include_output and "output" in examples:
        outputs = examples["output"]
        output_ids = tokenizer(
            outputs, max_length=1024, truncation=True, padding="max_length"
        ).input_ids
        model_inputs["output_ids"] = output_ids  # Keep outputs for downstream use

    # Keep idx for alignment
    if "idx" in examples:
        model_inputs["idx"] = examples["idx"]

    return model_inputs

# Load and preprocess the dataset
dataset = load_dataset("json", data_files={
    "train": os.path.join(dataset_path, "train.jsonl"),
    "test": os.path.join(dataset_path, "test.jsonl"),
    "result": os.path.join(dataset_path, "result.jsonl")
})

# Preprocess the dataset
def preprocess_dataset(split, include_output=False):
    return dataset[split].map(
        lambda x: preprocess_function(x, include_output),
        batched=True,
        remove_columns=[col for col in dataset[split].column_names if col not in ["idx", "inputs", "target", "output"]]
    )

# Tokenize datasets, ensuring that output is only included in the train split
tokenized_datasets = {
    "train": preprocess_dataset("train", include_output=True),
    "test": preprocess_dataset("test", include_output=False),
    "result": preprocess_dataset("result", include_output=False),
}

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./T5_D2N_Model",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    per_device_train_batch_size=8,  # Optimized for RTX 3090
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Adjust for memory efficiency
    num_train_epochs=10,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # Mixed precision for faster training
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=False,  # Preserve alignment fields like idx
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
trainer.save_model("./T5_D2N_Model")
tokenizer.save_pretrained("./T5_D2N_Model")

# Analyze the result.jsonl (Post-Training Evaluation)
result_dataset = tokenized_datasets["result"]
print("First example from result.jsonl (processed):", result_dataset[0])
