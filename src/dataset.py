import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os

def create_dataset(samples_per_class=None):
    """
    Create train/val/test splits from balanced dataset.

    Args:
        samples_per_class: Number of samples to use per class (None = use all data).
                          E.g., 8333 will give ~25k total, 16666 will give ~50k total and 82503 is full dataset.
    """
    data = pd.read_parquet("data/balanced_dataset.parquet")

    # Optionally downsample to reduce training time
    if samples_per_class is not None:
        downsampled_data = []
        for label in [0, 1, 2]:
            label_data = data[data['label'] == label].sample(n=samples_per_class, random_state=42)
            downsampled_data.append(label_data)
        data = pd.concat(downsampled_data, ignore_index=True).sample(frac=1, random_state=42)  # Shuffle
        print(f"Downsampled to {len(data)} total samples ({samples_per_class} per class)")

    train_val, test = train_test_split(data, test_size=0.1, random_state=42, shuffle=True, stratify=data['label'])

    train, val = train_test_split(train_val, test_size=0.1111, random_state=42, shuffle=True, stratify=train_val['label'])

    os.makedirs("dataset", exist_ok=True)
    train.to_parquet("datasets/train.parquet")
    val.to_parquet("datasets/validate.parquet")
    test.to_parquet("datasets/test.parquet")

class PostpartumDataset(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        self.data = data
        self.texts = data["selftext"].tolist()
        self.labels = data["label"].tolist()
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = int(self.labels[index])
        tokenized = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": label
        }

    def __len__(self) -> int:
        return len(self.texts)


def get_datasets(tokenizer):
    """
    Load train, validation, and test datasets.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = PostpartumDataset(pd.read_parquet("datasets/train.parquet"), tokenizer=tokenizer)
    val_dataset = PostpartumDataset(pd.read_parquet("datasets/validate.parquet"), tokenizer=tokenizer)
    test_dataset = PostpartumDataset(pd.read_parquet("datasets/test.parquet"), tokenizer=tokenizer)

    return train_dataset, val_dataset, test_dataset
    
