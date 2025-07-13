#!/usr/bin/env python3
"""
Nexus Model Training Script - Optimized for Fast Training
Trains a custom AI model for code generation and conversation
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from tqdm import tqdm
from torch.optim import AdamW

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CodeDataset(Dataset):
    """Dataset for code training data."""
    
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 256):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Processing {len(data)} training examples...")
        for item in tqdm(data, desc="Tokenizing data"):
            # Format the training example
            if "prompt" in item and "completion" in item:
                # Instruction-following format
                text = f"Instruction: {item['prompt']}\nCode: {item['completion']}\n"
            elif "code" in item:
                # Raw code format - limit length for faster processing
                text = item["code"][:1000]  # Limit code length
            else:
                continue
            
            # Tokenize the text
            encoding = self.tokenizer.encode(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            self.examples.append(encoding.squeeze())
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def load_training_data(sources: List[str], max_samples: int = 1000) -> List[Dict[str, str]]:
    """Load training data from specified sources."""
    all_data = []
    
    for source in sources:
        if source == "custom":
            # Load custom training data
            custom_file = "data/custom_training_data.json"
            if os.path.exists(custom_file):
                with open(custom_file, 'r') as f:
                    custom_data = json.load(f)
                all_data.extend(custom_data[:max_samples])
                logging.info(f"Loaded {len(custom_data[:max_samples])} examples from custom data")
            else:
                logging.warning("Custom training data not found, creating sample data...")
                create_training_data()
                with open(custom_file, 'r') as f:
                    custom_data = json.load(f)
                all_data.extend(custom_data[:max_samples])
                logging.info(f"Loaded {len(custom_data[:max_samples])} examples from custom data")
        
        elif source == "codealpaca":
            # Load CodeAlpaca dataset
            try:
                from datasets import load_dataset
                dataset = load_dataset("sahil2801/CodeAlpaca-20k")
                
                # Convert to our format
                codealpaca_data = []
                for item in dataset['train']:
                    if item.get('prompt') and item.get('completion'):
                        codealpaca_data.append({
                            'prompt': item['prompt'],
                            'completion': item['completion']
                        })
                
                all_data.extend(codealpaca_data[:max_samples])
                logging.info(f"Loaded {len(codealpaca_data[:max_samples])} examples from Code Alpaca")
                
            except Exception as e:
                logging.warning(f"Failed to load CodeAlpaca: {e}")
        
        elif source == "codeparrot":
            # Load CodeParrot dataset
            try:
                from datasets import load_dataset
                dataset = load_dataset("codeparrot/github-code", streaming=True)
                
                # Convert to our format
                codeparrot_data = []
                for i, item in enumerate(dataset['train']):
                    if i >= max_samples:
                        break
                    if item.get('code'):
                        # Create a simple prompt for code generation
                        codeparrot_data.append({
                            'prompt': f"Generate Python code for: {item.get('repo_name', 'a program')}",
                            'completion': item['code'][:500]  # Limit code length
                        })
                
                all_data.extend(codeparrot_data)
                logging.info(f"Loaded {len(codeparrot_data)} examples from CodeParrot")
                
            except Exception as e:
                logging.warning(f"Failed to load CodeParrot: {e}")
    
    logging.info(f"Loaded {len(all_data)} training examples")
    return all_data

def create_training_data():
    """Create sample training data for the model."""
    sample_data = [
        # Code generation examples
        {
            "prompt": "Create a function to calculate the factorial of a number",
            "completion": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        },
        {
            "prompt": "Write a function to check if a string is a palindrome",
            "completion": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"
        },
        {
            "prompt": "Create a class for a simple calculator",
            "completion": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError('Cannot divide by zero')\n        return a / b"
        },
        {
            "prompt": "Write a function to find the maximum element in a list",
            "completion": "def find_max(lst):\n    if not lst:\n        return None\n    return max(lst)"
        },
        {
            "prompt": "Create a function to reverse a string",
            "completion": "def reverse_string(s):\n    return s[::-1]"
        },
        {
            "prompt": "Write a function to check if a number is prime",
            "completion": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
        },
        {
            "prompt": "Create a function to generate Fibonacci sequence",
            "completion": "def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib"
        },
        {
            "prompt": "Write a function to sort a list using bubble sort",
            "completion": "def bubble_sort(lst):\n    n = len(lst)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if lst[j] > lst[j+1]:\n                lst[j], lst[j+1] = lst[j+1], lst[j]\n    return lst"
        },
        {
            "prompt": "Create a function to count vowels in a string",
            "completion": "def count_vowels(s):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in s if char in vowels)"
        },
        {
            "prompt": "Write a function to check if two strings are anagrams",
            "completion": "def are_anagrams(s1, s2):\n    return sorted(s1.lower()) == sorted(s2.lower())"
        },
        # Conversation examples
        {
            "prompt": "Hello, how are you?",
            "completion": "Hello! I'm doing well, thank you for asking. I'm here to help you with coding tasks and programming questions. How can I assist you today?"
        },
        {
            "prompt": "What can you help me with?",
            "completion": "I can help you with many programming tasks! I can generate code from natural language descriptions, analyze existing code, help debug issues, explain programming concepts, and assist with file operations. Just let me know what you need!"
        },
        {
            "prompt": "Can you write code?",
            "completion": "Yes, I can generate code in multiple programming languages including Python, JavaScript, Java, C++, and more. Just describe what you want the code to do, and I'll create it for you. For example, you could say 'create a function to calculate fibonacci numbers' or 'write a class for a todo list'."
        },
        {
            "prompt": "How do I create a web server?",
            "completion": "I can help you create a web server! Here are a few common approaches:\n\n1. Python with Flask:\n```python\nfrom flask import Flask\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello, World!'\n\nif __name__ == '__main__':\n    app.run(debug=True)\n```\n\n2. Python with FastAPI:\n```python\nfrom fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/')\ndef read_root():\n    return {'Hello': 'World'}\n```\n\nWould you like me to generate a complete web server example for you?"
        },
        {
            "prompt": "What's the difference between a list and a tuple?",
            "completion": "Great question! Here are the key differences between lists and tuples in Python:\n\n**Lists:**\n- Mutable (can be changed after creation)\n- Use square brackets: [1, 2, 3]\n- Can add, remove, or modify elements\n- Generally used for collections that might change\n\n**Tuples:**\n- Immutable (cannot be changed after creation)\n- Use parentheses: (1, 2, 3)\n- Cannot add, remove, or modify elements\n- Generally used for fixed collections\n- Slightly more memory efficient\n\nExample:\n```python\nmy_list = [1, 2, 3]\nmy_list.append(4)  # Works\n\nmy_tuple = (1, 2, 3)\nmy_tuple.append(4)  # Error!\n```"
        },
        {
            "prompt": "Help me understand decorators",
            "completion": "Decorators are a powerful Python feature that allow you to modify or enhance functions. Here's a simple explanation:\n\nA decorator is a function that takes another function and extends its behavior without explicitly modifying it.\n\nBasic example:\n```python\ndef my_decorator(func):\n    def wrapper():\n        print('Something before the function')\n        func()\n        print('Something after the function')\n    return wrapper\n\n@my_decorator\ndef say_hello():\n    print('Hello!')\n\nsay_hello()\n```\n\nThis will output:\nSomething before the function\nHello!\nSomething after the function\n\nDecorators are commonly used for logging, timing, authentication, and more!"
        },
        {
            "prompt": "How do I install packages in Python?",
            "completion": "There are several ways to install Python packages:\n\n1. **pip** (most common):\n```bash\npip install package_name\n```\n\n2. **Using requirements.txt**:\n```bash\npip install -r requirements.txt\n```\n\n3. **For specific versions**:\n```bash\npip install package_name==1.2.3\n```\n\n4. **In a virtual environment** (recommended):\n```bash\npython -m venv myenv\nsource myenv/bin/activate  # On Windows: myenv\\Scripts\\activate\npip install package_name\n```\n\n5. **Using conda** (if you use Anaconda):\n```bash\nconda install package_name\n```\n\nAlways use virtual environments to avoid conflicts between projects!"
        }
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save sample data
    with open("data/custom_training_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logging.info(f"Created {len(sample_data)} sample training examples")

def train_model(
    model_name: str = "microsoft/DialoGPT-small",  # Use smaller model for faster training
    output_dir: str = "./model/nexus_model",
    epochs: int = 2,  # Reduced epochs for faster training
    batch_size: int = 4,  # Increased batch size for CPU
    learning_rate: float = 5e-5,
    max_length: int = 256,  # Reduced sequence length
    data_sources: List[str] = None,
    max_samples_per_source: int = 500  # Limit samples for faster training
):
    """Train the Nexus model with optimizations for faster training."""
    
    if data_sources is None:
        data_sources = ["custom"]
    
    # Create sample data if it doesn't exist
    if "custom" in data_sources and not os.path.exists("data/custom_training_data.json"):
        create_training_data()
    
    # Load tokenizer and model
    logging.info(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load training data with size limits
    logging.info("Loading training data...")
    training_data = load_training_data(data_sources, max_samples_per_source)
    
    if not training_data:
        logging.error("No training data loaded!")
        return
    
    logging.info(f"Loaded {len(training_data)} training examples")
    
    # Create dataset
    dataset = CodeDataset(training_data, tokenizer, max_length)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Setup training arguments optimized for CPU speed
training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,  # Save more frequently
        save_total_limit=2,
        prediction_loss_only=True,
        learning_rate=learning_rate,
        logging_steps=50,  # Log more frequently
        save_strategy="steps",
        load_best_model_at_end=False,
        warmup_steps=50,  # Reduced warmup
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        # CPU optimizations
        dataloader_num_workers=0,  # Single worker for CPU
        remove_unused_columns=False,
        # Mixed precision for CPU (if supported)
        fp16=False,  # Disable for CPU
        # Gradient accumulation for larger effective batch size
        gradient_accumulation_steps=2,
    )
    
    # Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
)

    # Train the model
    logging.info("Starting training...")
    print(f"Training on {len(dataset)} examples for {epochs} epochs")
    print(f"Batch size: {batch_size}, Sequence length: {max_length}")
    trainer.train()
    
    # Save the model and tokenizer
    logging.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save training config
    config = {
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "data_sources": data_sources,
        "training_examples": len(training_data),
        "max_samples_per_source": max_samples_per_source
    }
    
    with open(f"{output_dir}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info("Training completed successfully!")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Nexus AI model")
    parser.add_argument("--sources", nargs="+", default=["custom"], 
                       help="Training data sources (custom, codealpaca, codeparrot)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples per source")
    parser.add_argument("--model-name", type=str, default="microsoft/DialoGPT-small", 
                       help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="./model/nexus_model", 
                       help="Output directory for trained model")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Fast Training Mode - Optimized for CPU")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples per source: {args.max_samples}")
    print(f"Sequence length: {args.max_length}")
    
    try:
        # Load base model
        logging.info(f"Loading base model: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load training data
        logging.info("Loading training data...")
        training_data = load_training_data(args.sources, args.max_samples)
        
        if not training_data:
            logging.error("No training data loaded!")
            return
        
        print(f"Processing {len(training_data)} training examples...")
        
        # Prepare dataset
        dataset = CodeDataset(training_data, tokenizer, args.max_length)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()
        
        optimizer = AdamW(model.parameters(), lr=5e-5)
        
        # Training loop
        logging.info("Starting training...")
        print(f"Training on {len(training_data)} examples for {args.epochs} epochs")
        print(f"Batch size: {args.batch_size}, Sequence length: {args.max_length}")
        
        for epoch in range(args.epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
                batch = batch.to(device)
                
                # Forward pass
                outputs = model(batch, labels=batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, Avg Loss: {avg_loss:.4f}")
        
        # Save model
        os.makedirs(args.output_dir, exist_ok=True)
        logging.info(f"Saving model to {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        logging.info("Training completed successfully!")
        print("✅ Training completed! Model saved to:", args.output_dir)
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
