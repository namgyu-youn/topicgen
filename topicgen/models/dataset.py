
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TopicDataset(Dataset):
    """PyTorch dataset for topic classification."""

    def __init__(self,
                texts: list[str],
                labels: np.ndarray,
                tokenizer_name: str = "bert-base-uncased",
                max_length: int = 512):
        """
        Initialize the topic dataset.

        Args:
            texts: List of text content
            labels: Array of multi-hot encoded labels
            tokenizer_name: Hugging Face tokenizer name
            max_length: Maximum sequence length for tokenization
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        """Get dataset size."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Get a dataset item by index."""
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Remove batch dimension from tokenizer output
        item = {key: val.squeeze(0) for key, val in encoding.items()}

        # Add label
        item["labels"] = torch.tensor(label, dtype=torch.float)

        return item


class TopicDataCollator:
    """Collator for batching topic classification data."""

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            features: List of tokenized examples

        Returns:
            Batch dictionary with tensors
        """
        # Stack input tensors
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features])
        }

        # Add token_type_ids if present
        if "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.stack([f["token_type_ids"] for f in features])

        return batch
