import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Define the paths
dataset_path = ""

# Load the tokenizer and model (using T5-base)
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Define a preprocessing function
def preprocess_function(examples):
    inputs = examples["inputs"]
    targets = examples["target"]

    # Tokenize inputs and targets
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="longest"  # Changed to 'longest'
    )
    labels = tokenizer(
        targets, max_length=512, truncation=True, padding="longest"  # Changed to 'longest'
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

# Define the training arguments (adjusted for T5-base)
training_args = TrainingArguments(
    output_dir="./T5_D2N_Model",         # Output directory for model checkpoints
    evaluation_strategy="steps",         # Evaluate every `eval_steps`
    eval_steps=62,                       # Evaluate after every 62 steps (~620 steps for 10 epochs)
    save_steps=1240,                     # Save checkpoints after every 62 steps
    logging_steps=50,                    # Log metrics every 50 steps
    per_device_train_batch_size=8,       # Reduced batch size for T5-base
    per_device_eval_batch_size=8,        # Match eval batch size
    gradient_accumulation_steps=4,       # Keep this for efficient batch size
    num_train_epochs=10,                 # Number of training epochs
    learning_rate=3e-5,                  # Learning rate optimized for T5-base
    weight_decay=0.01,                   # Regularization to avoid overfitting
    save_total_limit=2,                  # Limit number of saved checkpoints
    fp16=True,                           # Mixed precision for faster training
    load_best_model_at_end=True,         # Load the best model based on eval performance
    report_to="none",                    # Avoid reporting to external tools
    remove_unused_columns=True,          # Preserve alignment fields like 'idx'
    warmup_steps=500,                    # Added warm-up steps
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
