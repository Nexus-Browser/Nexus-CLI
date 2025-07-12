from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# Load the Code Alpaca dataset
dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train")

# Create an iterator to feed the tokenizer
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        # Combine instruction and response for tokenizer training
        yield [prompt + completion for prompt, completion in zip(batch["prompt"], batch["completion"])]

# Initialize and train the tokenizer
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=10000,  # Increased vocab size for a more general model
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
)

# Save the tokenizer
tokenizer.save_model("tokenizer")

print("Tokenizer trained and saved.")
