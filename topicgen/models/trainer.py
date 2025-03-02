import logging
import os
from typing import Any

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from .dataset import TopicDataCollator, TopicDataset

logger = logging.getLogger(__name__)

class TopicTrainer:
    """Trainer for multi-label topic classification."""

    def __init__(self,
                 base_model: str = "bert-base-uncased",
                 num_topics: int = 100,
                 learning_rate: float = 5e-5,
                 batch_size: int = 16,
                 num_epochs: int = 5,
                 device: str | None = None):
        """
        Initialize the topic trainer.

        Args:
            base_model: Base transformer model name
            num_topics: Number of topics to classify
            learning_rate: Learning rate for optimization
            batch_size: Training batch size
            num_epochs: Number of training epochs
            device: PyTorch device (auto-detected if None)
        """
        self.base_model = base_model
        self.num_topics = num_topics
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the classification model."""
        try:
            # Configure model for multi-label classification
            config = AutoConfig.from_pretrained(
                self.base_model,
                num_labels=self.num_topics,
                problem_type="multi_label_classification"
            )

            # Load pre-trained model with custom config
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model,
                config=config
            )

            # Move model to device
            self.model.to(self.device)

            logger.info(f"Initialized model {self.base_model} for {self.num_topics} topics")

        except Exception as e:
            logger.error(f"Error initializing model: {e!s}")
            raise

    def _create_dataloaders(self, data_splits: dict[str, dict[str, Any]]) -> dict[str, DataLoader]:
        """
        Create DataLoaders for training, validation, and testing.

        Args:
            data_splits: Dictionary containing train/val/test splits

        Returns:
            Dictionary of DataLoaders
        """
        dataloaders = {}
        collator = TopicDataCollator()

        for split, split_data in data_splits.items():
            # Create dataset
            dataset = TopicDataset(
                texts=split_data["texts"],
                labels=split_data["labels"],
                tokenizer_name=self.base_model
            )

            # Create dataloader
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(split == "train"),
                collate_fn=collator,
                num_workers=min(4, os.cpu_count() or 1)
            )

        return dataloaders

    def train(self, data_splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """
        Train the topic classification model.

        Args:
            data_splits: Dictionary containing train/val/test splits

        Returns:
            Dictionary with training results and metrics
        """
        # Create dataloaders
        dataloaders = self._create_dataloaders(data_splits)

        # Prepare optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )

        # Prepare scheduler
        total_steps = len(dataloaders["train"]) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        logger.info(f"Starting training for {self.num_epochs} epochs")
        best_val_loss = float('inf')
        best_model_state = None
        training_stats = []

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(dataloaders["train"], desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(dataloaders["train"])

            # Validation
            val_metrics = self._evaluate(dataloaders["val"])

            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val F1: {val_metrics['f1']:.4f}")

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics["loss"],
                    'val_f1': val_metrics["f1"],
                }
                logger.info(f"New best model saved (F1: {val_metrics['f1']:.4f})")

            # Store stats
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': val_metrics["loss"],
                'val_precision': val_metrics["precision"],
                'val_recall': val_metrics["recall"],
                'val_f1': val_metrics["f1"],
            }
            training_stats.append(epoch_stats)

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state["model_state_dict"])
            logger.info(f"Loaded best model from epoch {best_model_state['epoch']}")

        # Test on held-out test set
        test_metrics = self._evaluate(dataloaders["test"])
        logger.info(f"Test metrics - "
                   f"Loss: {test_metrics['loss']:.4f}, "
                   f"F1: {test_metrics['f1']:.4f}, "
                   f"Precision: {test_metrics['precision']:.4f}, "
                   f"Recall: {test_metrics['recall']:.4f}")

        return {
            "training_stats": training_stats,
            "best_model_state": best_model_state,
            "test_metrics": test_metrics
        }

    def _evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate model on a dataloader.

        Args:
            dataloader: DataLoader for evaluation

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                # Apply sigmoid for multi-label prediction
                predictions = torch.sigmoid(logits).cpu().numpy()
                labels = batch["labels"].cpu().numpy()

                # Store results
                total_loss += loss.item()
                all_preds.append(predictions)
                all_labels.append(labels)

        # Concatenate batch results
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Convert predictions to binary
        binary_preds = (all_preds > 0.5).astype(float)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, binary_preds,
            average='micro',  # Use micro-average for multi-label
            zero_division=0
        )

        avg_loss = total_loss / len(dataloader)

        return {
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def save_model(self, path: str):
        """
        Save the trained model.

        Args:
            path: Directory to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Save model state
        model_path = os.path.join(path, "model.pt")
        torch.save(self.model.state_dict(), model_path)

        # Save configuration
        self.model.config.save_pretrained(path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a trained model.

        Args:
            path: Path to the saved model
        """
        # Load model state
        model_path = os.path.join(path, "model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        logger.info(f"Model loaded from {path}")