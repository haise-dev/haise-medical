import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq

# Define the dataset paths
dataset_path = ""

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-large")
model = T5ForConditionalGeneration.from_pretrained("t5-large")

# Load the dataset
dataset = load_dataset("json", data_files={
    "train": os.path.join(dataset_path, "train.jsonl"),
    "test": os.path.join(dataset_path, "test.jsonl"),
    "result": os.path.join(dataset_path, "result.jsonl"),
})

# Tokenization function
def tokenize_function(examples):
    model_inputs = tokenizer(examples['inputs'], max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], max_length=512, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Apply tokenization to all splits
dataset = dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Data collator for Seq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the training arguments for D2N
training_args = TrainingArguments(
    output_dir="./T5_D2N_Model",         # Output directory for checkpoints and logs
    evaluation_strategy="steps",         # Evaluate at regular steps
    eval_steps=1,                        # Evaluate every 1 steps
    save_steps=100,                      # Save checkpoint every 100 steps
    logging_steps=2,                     # Log training metrics every 2 steps
    per_device_train_batch_size=4,       # Reduce batch size to fit T5-large on GPU
    per_device_eval_batch_size=4,        # Match batch size for evaluation
    gradient_accumulation_steps=8,       # Effective batch size = 4 * 8 = 32
    num_train_epochs=10,                 # Train for 10 epochs
    learning_rate=1e-4,                  # Lower learning rate for T5-large
    weight_decay=0.01,                   # Regularization to avoid overfitting
    save_total_limit=2,                  # Keep the last 2 checkpoints
    fp16=True,                           # Mixed precision for memory efficiency
    load_best_model_at_end=True,         # Automatically load the best model
    warmup_steps=50,                     # Shorter warm-up since training is smaller
    report_to="none",                    # Avoid reporting to external tools
    remove_unused_columns=False          # Preserve alignment fields like 'idx'
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["result"],      # Use 'result.jsonl' for evaluation
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
trainer.save_model("./T5_D2N_Model")
tokenizer.save_pretrained("./T5_D2N_Model")

# Evaluate the model on the test dataset
test_results = trainer.evaluate(dataset["test"])
print("Test Results:", test_results)
