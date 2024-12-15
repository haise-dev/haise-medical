import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Define the paths
dataset_path = ""

# Load the tokenizer and model
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define a preprocessing function
def preprocess_function(examples):
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

    return model_inputs

# Load the dataset
dataset = load_dataset("json", data_files={
    "train": os.path.join(dataset_path, "train.jsonl"),
    "test": os.path.join(dataset_path, "test.jsonl"),
    "result": os.path.join(dataset_path, "result.jsonl"),
})

# Preprocess the dataset
def preprocess_dataset(split):
    return dataset[split].map(
        preprocess_function,
        batched=True,
        remove_columns=["output"]  # Explicitly remove unwanted columns
    )

# Tokenize datasets
tokenized_datasets = {
    "train": preprocess_dataset("train"),
    "test": preprocess_dataset("test"),
    "result": preprocess_dataset("result"),
}

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./T5_D2N_Model",
    evaluation_strategy="steps",
    eval_steps=500,  # Adjust based on dataset size
    save_steps=500,  # Save less frequently to avoid overhead
    logging_steps=100,
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=3e-5,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # Use mixed precision if supported
    load_best_model_at_end=True,
    report_to="none",
    remove_unused_columns=True,  # Ensures compatibility with the Trainer
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
