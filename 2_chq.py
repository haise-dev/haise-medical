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
    "validate": os.path.join(dataset_path, "validate.jsonl"),
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

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./T5_CHQ_Model",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    logging_steps=100,
    per_device_train_batch_size=8,  # Optimized for RTX 3090
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # Adjust for memory efficiency
    num_train_epochs=10,  # Adjust based on the dataset size
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
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validate"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
trainer.save_model("./T5_CHQ_Model")
tokenizer.save_pretrained("./T5_CHQ_Model")

# Evaluate the model on the test dataset
test_results = trainer.evaluate(dataset["test"])
print(test_results)
