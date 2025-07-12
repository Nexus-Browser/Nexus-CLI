import json, torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset

# Load the configuration (though some parts might be overridden by pre-trained model config)
with open("model_config.json") as f:
    config = json.load(f)

# Load pre-trained GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Add a padding token if it doesn't exist and resize model embeddings
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Update config with actual vocab size from the loaded tokenizer
config["vocab_size"] = len(tokenizer)

class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, block_size):
        self.examples = []
        for example in dataset:
            # Combine prompt and completion for training
            # Add a special token to separate prompt and completion if desired, e.g., <|endoftext|>
            # For simplicity, we'll just concatenate for now.
            full_text = example["prompt"] + example["completion"]

            # Encode the full text
            encoded_text = tokenizer.encode(full_text, truncation=True, max_length=block_size)

            # Pad or truncate to the block size
            if len(encoded_text) < block_size:
                # Pad with the pad_token_id
                padded_text = encoded_text + [tokenizer.pad_token_id] * (block_size - len(encoded_text))
            else:
                # Truncate
                padded_text = encoded_text[:block_size]

            self.examples.append(padded_text)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5) # Standard learning rate for fine-tuning

# Load a smaller subset of the Code Alpaca dataset for faster training
alpaca_dataset = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train[:1000]")
dataset = InstructionDataset(alpaca_dataset, tokenizer, config["n_positions"])
loader = DataLoader(dataset, batch_size=4)

# Fine-tune the model
model.train() # Set model to training mode
for epoch in range(1): # Train for only 1 epoch for speed
    for batch in loader:
        inputs = batch.to(device)
        outputs = model(inputs, labels=inputs) # labels=inputs for causal language modeling
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Loss: {loss.item():.4f}")

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_nexus_model")
tokenizer.save_pretrained("./fine_tuned_nexus_model")

print("Fine-tuned model and tokenizer saved to ./fine_tuned_nexus_model")