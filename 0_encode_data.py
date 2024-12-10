import os
import jsonlines

# Define the folder paths
raw_data_path = ' '
output_data_path = ' '

# Function to process chq dataset
def process_chq(data):
    # Replace placeholders with special tokens
    data = data.replace("[DATE]", "<DATE>")
    data = data.replace("[NAME]", "<NAME>")
    data = data.replace("[LOCATION]", "<LOCATION>")
    data = data.replace("[CONTACT]", "<CONTACT>")
    data = data.replace("SUBJECT:", "<SUBJECT>")
    data = data.replace("MESSAGE:", "<MESSAGE>")
    return data

# Function to process d2n dataset
def process_d2n(data):
    # Replace doctor and patient markers
    data = data.replace("[doctor]", "<DOCTOR>")
    data = data.replace("[patient]", "<PATIENT>")
    return data

# Function to process opi dataset (simple inputs/targets)
def process_opi(inputs, target):
    # In this case, no specific placeholder replacements are needed
    return inputs, target

# Function to save processed data to a new folder
def save_to_file(output_file, processed_data):
    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(processed_data)

# Function to process the dataset and save to a new folder
def process_and_save_dataset(dataset_type, dataset_name):
    input_folder = os.path.join(raw_data_path, dataset_type)
    output_folder = os.path.join(output_data_path, dataset_type)
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.jsonl'):
            input_file = os.path.join(input_folder, file_name)
            output_file = os.path.join(output_folder, file_name)

            # Open and read the dataset file
            with jsonlines.open(input_file) as reader:
                processed_data = []
                for obj in reader:
                    idx = obj.get("idx")  # Ensure idx is preserved
                    # Process the datasets based on their type
                    if dataset_type == 'chq':
                        inputs = process_chq(obj['inputs'])
                        target = obj['target']
                        processed_data.append({"idx": idx, "inputs": inputs, "target": target})
                    elif dataset_type == 'd2n':
                        inputs = process_d2n(obj['inputs'])
                        target = obj['target']
                        # Check if 'output' exists (only for 'train.jsonl')
                        if 'output' in obj:
                            output = obj['output']
                            processed_data.append({"idx": idx, "inputs": inputs, "target": target, "output": output})
                        else:
                            processed_data.append({"idx": idx, "inputs": inputs, "target": target})
                    elif dataset_type == 'opi':
                        inputs, target = process_opi(obj['inputs'], obj['target'])
                        # Check if 'output' exists (only for 'train.jsonl')
                        if 'output' in obj:
                            output = obj['output']
                            processed_data.append({"idx": idx, "inputs": inputs, "target": target, "output": output})
                        else:
                            processed_data.append({"idx": idx, "inputs": inputs, "target": target})

            # Save the processed data to the output folder
            save_to_file(output_file, processed_data)

# Process all datasets
for dataset_type in ['chq', 'd2n', 'opi']:
    process_and_save_dataset(dataset_type, raw_data_path)

print("Processing completed and data saved.")
