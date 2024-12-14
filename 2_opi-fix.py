import os
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

# Paths to the dataset
dataset_path = ''  # Provide the actual path to your dataset

# Load the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Load dataset and ensure consistent columns
columns_to_keep = ['idx', 'inputs', 'target']  # Columns common to all splits
opi_dataset = load_dataset('json', data_files={
    'train': os.path.join(dataset_path, 'train.jsonl'),
    'test': os.path.join(dataset_path, 'test.jsonl'),
    'validate': os.path.join(dataset_path, 'validate.jsonl'),
    'result': os.path.join(dataset_path, 'result.jsonl'),
})

# Define tokenization function
def tokenize_function(examples):
    # Tokenize inputs and targets
    model_inputs = tokenizer(examples['inputs'], max_length=512, padding='max_length', truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], max_length=512, padding='max_length', truncation=True)
    
    model_inputs['labels'] = labels['input_ids']

    return model_inputs

# Process datasets
for split in ['train', 'test', 'validate', 'result']:
    opi_dataset[split] = opi_dataset[split].map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[col for col in opi_dataset[split].column_names if col not in columns_to_keep]
    )

# Set format for PyTorch
opi_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Data Collator for Seq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./T5_OPI_Model",
    evaluation_strategy="steps",
    save_steps=8000,
    eval_steps=200,
    logging_dir="./logs",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    logging_steps=500,
    weight_decay=0.01,
    learning_rate=3e-5,
    lr_scheduler_type="linear",
    fp16=True,
    load_best_model_at_end=True,
)

# Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=opi_dataset['train'],
    eval_dataset=opi_dataset['validate'],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the final model and tokenizer
trainer.save_model("./T5_OPI_Model")
tokenizer.save_pretrained("./T5_OPI_Model")
