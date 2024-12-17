import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Define paths to the models (you can replace these with actual model paths)
model_paths = {
    "d2n": "",
    "chq": "",
    "opi": ""
}

# Function to check if model path is available
def is_model_available(model_path):
    return os.path.isdir(model_path)

# Check which models are available
available_models = {name: path for name, path in model_paths.items() if is_model_available(path)}

# Load the tokenizers and models for available models
models = {name: T5ForConditionalGeneration.from_pretrained(path) for name, path in available_models.items()}
tokenizers = {name: T5Tokenizer.from_pretrained(path) for name, path in available_models.items()}

# Define the summarization function
def summarize_dialogue(dialogue: str, model_name: str, max_length: int = 100, min_length: int = 5) -> str:
    """
    Summarizes a dialogue using the selected T5 model.

    Args:
        dialogue (str): The input dialogue text to summarize.
        model_name (str): The name of the model to use for summarization (e.g., 'd2n', 'chq', 'opi').
        max_length (int): The maximum length of the summary.
        min_length (int): The minimum length of the summary.

    Returns:
        str: The generated summary.
    """
    tokenizer = tokenizers[model_name]
    model = models[model_name]

    # Ensure the dialogue does not exceed the max token length
    chunk_size = 1024  # Maximum number of tokens that can be handled at once
    tokens = tokenizer.encode(dialogue, return_tensors="pt", truncation=True, max_length=chunk_size)

    # Split into smaller chunks if the length exceeds max_length
    dialogue_chunks = [tokens[0][i:i+chunk_size] for i in range(0, len(tokens[0]), chunk_size)]

    summaries = []
    for chunk in dialogue_chunks:
        # Generate the summary for each chunk
        summary_ids = model.generate(
            chunk.unsqueeze(0),
            max_length=max_length,
            min_length=min_length,
            no_repeat_ngram_size=2,  # Prevent repetition of n-grams
            temperature=0.7,  # Controls randomness: lower is more deterministic
            top_p=0.9,  # Controls diversity: only the top p probability mass is considered
            num_beams=4,  # Beam search to improve quality
            early_stopping=True
        )
        # Decode and store the summary
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    # Combine summaries from each chunk
    final_summary = " ".join(summaries)
    return final_summary

# Interactive testing
if __name__ == "__main__":
    print("Clinical Text Summarization Interface")

    if available_models:
        print("Available models:")
        for model in available_models:
            print(f"- {model}")
    else:
        print("No models are available in the specified paths.")
        exit(1)

    print("Type 'exit' to quit.\n")

    while True:
        # Ask the user to choose a model
        model_choice = input("Enter model name ('d2n', 'chq', 'opi') or 'exit' to quit: ").strip().lower()
        if model_choice == "exit":
            print("Exiting...")
            break
        if model_choice not in available_models:
            print(f"Model '{model_choice}' is not available. Please choose one of the available models.")
            continue

        # Input dialogue text
        user_input = input("Enter a dialogue (up to 1000 words): ").strip()
        if not user_input:
            print("Please enter a valid dialogue!")
            continue

        print("\nSummarizing...")
        try:
            result = summarize_dialogue(user_input, model_choice)
            print("Summary:\n", result)
        except Exception as e:
            print(f"Error during summarization: {e}")
        print("-" * 50)
