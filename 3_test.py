import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the trained model and tokenizer
model_path = ""  # Replace with the path to your trained model
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# Define a function for generating output
def generate_output(input_text, min_length=5, max_length=100):
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Generate the output with specified constraints
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            min_length=min_length,
            max_length=max_length,
            num_beams=4,  # Beam search for better results
            early_stopping=True
        )
    
    # Decode the output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# Interface for testing
def main():
    print("===== T5 Model Interface =====")
    print("Type 'exit' to quit the interface.")
    print("-----------------------------------")
    
    while True:
        # Get input text from the user
        input_text = input("Enter input text: ").strip()
        
        if input_text.lower() == "exit":
            print("Exiting... Goodbye!")
            break
        
        if not input_text:
            print("Input cannot be empty. Please try again.")
            continue
        
        # Generate and display output
        output = generate_output(input_text, min_length=5, max_length=100)
        print("\nGenerated Output:")
        print(output)
        print("-----------------------------------")

if __name__ == "__main__":
    main()
