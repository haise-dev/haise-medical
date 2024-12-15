import os
import json

def clean_jsonl_file_in_place(file_path):
    """
    Cleans a JSONL file in place by removing the 'output' column if it exists.
    
    :param file_path: Path to the JSONL file to be cleaned.
    """
    temp_file_path = file_path + ".tmp"
    
    with open(file_path, 'r', encoding='utf-8') as infile, open(temp_file_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            # Remove the 'output' column if it exists
            if "output" in data:
                del data["output"]
            # Write the cleaned data back to a temporary file
            json.dump(data, outfile)
            outfile.write('\n')
    
    # Replace the original file with the cleaned temporary file
    os.replace(temp_file_path, file_path)
    print(f"Cleaned file: {file_path}")

def preprocess_dataset_in_place(dataset_dir):
    """
    Preprocess all JSONL files in the dataset directory in place, removing the 'output' column.
    
    :param dataset_dir: Path to the dataset directory containing JSONL files.
    """
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                clean_jsonl_file_in_place(file_path)

# Set the dataset directory (adjust to your actual dataset path)
dataset_directory = "/path/to/your/dataset"

# Preprocess the dataset in place
preprocess_dataset_in_place(dataset_directory)
