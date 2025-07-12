from datasets import load_dataset

def get_codeparrot_dataset():
    # Load the CodeParrot dataset from Hugging Face
    dataset = load_dataset("codeparrot/codeparrot-clean-valid")
    
    # The dataset is already split into train and validation sets.
    # We can access them like this:
    train_dataset = dataset["train"]
    
    return train_dataset

if __name__ == "__main__":
    # This will download the dataset and print some info about it
    codeparrot_train = get_codeparrot_dataset()
    print(codeparrot_train)
