import os
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

# Paths to the dataset
dataset_path = ''  # Provide the actual path to your dataset

# Load the tokenizer and model (updated to T5-large)
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Load dataset (train, test, validate, and result files)
opi_dataset = load_dataset('json', data_files={
    'train': os.path.join(dataset_path, 'train.jsonl'),
    'test': os.path.join(dataset_path, 'test.jsonl'),
    'validate': os.path.join(dataset_path, 'validate.jsonl'),
    'result': os.path.join(dataset_path, 'result.jsonl'),
})

# Tokenize the dataset
def tokenize_function(examples, is_train=False):
    # Tokenize the inputs and targets
    model_inputs = tokenizer(examples['inputs'], max_length=512, padding='max_length', truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], max_length=512, padding='max_length', truncation=True)

    model_inputs['labels'] = labels['input_ids']

    # Handle the "output" field only in the training dataset
    if is_train:
        outputs = examples['output']  # Only available in train.jsonl
        output_ids = tokenizer(outputs, max_length=1024, padding='max_length', truncation=True)['input_ids']
        model_inputs['output_ids'] = output_ids

    # Always preserve idx for reference
    if 'idx' in examples:
        model_inputs['idx'] = examples['idx']

    return model_inputs

# Apply tokenization to each dataset, marking the 'train' dataset for handling 'output'
opi_dataset = opi_dataset.map(
    lambda x: tokenize_function(x, is_train=True),  # Use is_train=True for the 'train' split
    batched=True,
    num_proc=4,  # Parallelize processing (adjust based on your system)
    remove_columns=[col for col in opi_dataset['train'].column_names if col not in ['idx', 'inputs', 'target', 'output']]
)

# For the other datasets (test, validate, result), exclude 'output'
opi_dataset['test'] = opi_dataset['test'].map(
    lambda x: tokenize_function(x, is_train=False),
    batched=True,
    num_proc=4,
    remove_columns=[col for col in opi_dataset['test'].column_names if col not in ['idx', 'inputs', 'target']]
)

opi_dataset['validate'] = opi_dataset['validate'].map(
    lambda x: tokenize_function(x, is_train=False),
    batched=True,
    num_proc=4,
    remove_columns=[col for col in opi_dataset['validate'].column_names if col not in ['idx', 'inputs', 'target']]
)

opi_dataset['result'] = opi_dataset['result'].map(
    lambda x: tokenize_function(x, is_train=False),
    batched=True,
    num_proc=4,
    remove_columns=[col for col in opi_dataset['result'].column_names if col not in ['idx', 'inputs', 'target']]
)

# Set format for PyTorch
opi_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Data Collator for Seq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments (adjusted for T5-large and RTX 3090)
training_args = TrainingArguments(
    output_dir="./T5_OPI_Model",
    evaluation_strategy="steps",
    save_steps=2000,  # Save the model every 2000 steps
    eval_steps=500,   # Evaluate every 500 steps
    logging_dir="./logs",  # Log directory
    num_train_epochs=3,  # Adjust based on dataset size
    per_device_train_batch_size=2,  # Reduced batch size for T5-large
    per_device_eval_batch_size=2,  # Reduced batch size for evaluation
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch sizes
    logging_steps=500,  # Log every 500 steps
    weight_decay=0.01,
    learning_rate=3e-5,
    lr_scheduler_type="linear",  # Use linear learning rate scheduler
    fp16=True,  # Use mixed-precision training
    load_best_model_at_end=True,  # Load the best model at the end of training
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

# Start training
trainer.train()

# Save the final model
trainer.save_model("./T5_OPI_Model")

# Optionally, save the tokenizer
tokenizer.save_pretrained("./T5_OPI_Model")
