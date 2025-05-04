"""
Enhanced trainer with advanced training techniques and optimizations.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Tuple

import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from .dataset import TopicDataCollator, TopicDataset
from .training_optimizations import (
    OptimizationConfig,
    MixedPrecisionTrainer,
    DistributedTrainingManager,
    MemoryOptimizer,
    CheckpointManager
)
from .advanced_training import (
    FocalLoss,
    AsymmetricLoss,
    LabelSmoothingLoss,
    MixupAugmentation,
    AdversarialTraining,
    create_advanced_model
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTrainerConfig:
    """Configuration for the enhanced trainer."""

    # Model configuration
    base_model: str = "bert-base-uncased"
    num_topics: int = 100
    dropout_rate: float = 0.1

    # Training parameters
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    num_epochs: int = 5
    max_length: int = 512

    # Scheduler parameters
    scheduler_type: str = "linear"  # "linear" or "cosine"
    warmup_steps_fraction: float = 0.1

    # Loss function
    loss_type: str = "bce"  # "bce", "focal", "asymmetric", or "smoothing"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    asymmetric_gamma_pos: float = 1.0
    asymmetric_gamma_neg: float = 4.0
    smoothing_factor: float = 0.1

    # Advanced techniques
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_adversarial: bool = False
    adversarial_epsilon: float = 1e-2

    # Optimization techniques
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Device
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, OptimizationConfig):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result


class EnhancedTopicTrainer:
    """Enhanced trainer for multi-label topic classification with advanced techniques."""

    def __init__(self, config: EnhancedTrainerConfig):
        """
        Initialize the enhanced topic trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Auto-detect device if not specified
        if self.config.device is None:
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Using device: {self.config.device}")

        # Initialize optimization components
        self.mixed_precision = MixedPrecisionTrainer(
            enabled=self.config.optimization.use_mixed_precision
        )

        self.distributed_manager = DistributedTrainingManager(
            config=self.config.optimization
        )

        self.memory_optimizer = MemoryOptimizer(
            enabled=self.config.optimization.optimize_memory_usage
        )

        self.checkpoint_manager = CheckpointManager(
            config=self.config.optimization
        )

        # Initialize model
        self._initialize_model()

        # Initialize loss function
        self._initialize_loss_function()

        # Initialize mixup if enabled
        if self.config.use_mixup:
            self.mixup = MixupAugmentation(alpha=self.config.mixup_alpha)
        else:
            self.mixup = None

        # Initialize adversarial training if enabled
        if self.config.use_adversarial:
            self.adversarial = AdversarialTraining(
                model=self.model,
                epsilon=self.config.adversarial_epsilon
            )
        else:
            self.adversarial = None

        # Training metrics
        self.training_stats = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def _initialize_model(self):
        """Initialize the classification model with advanced configuration."""
        try:
            # Create advanced model
            self.model = create_advanced_model(
                base_model=self.config.base_model,
                num_topics=self.config.num_topics,
                dropout_rate=self.config.dropout_rate
            )

            # Apply memory optimizations
            self.model = self.memory_optimizer.optimize_model(self.model)

            # Set up distributed training if enabled
            if self.distributed_manager.setup():
                self.model = self.distributed_manager.prepare_model(self.model)
            else:
                # Move model to device
                self.model.to(self.config.device)

            logger.info(f"Initialized model {self.config.base_model} for {self.config.num_topics} topics")

        except Exception as e:
            logger.error(f"Error initializing model: {e!s}")
            raise

    def _initialize_loss_function(self):
        """Initialize the loss function based on configuration."""
        if self.config.loss_type == "focal":
            self.loss_fn = FocalLoss(
                gamma=self.config.focal_gamma,
                alpha=self.config.focal_alpha
            )
            logger.info(f"Using focal loss with gamma={self.config.focal_gamma}, alpha={self.config.focal_alpha}")

        elif self.config.loss_type == "asymmetric":
            self.loss_fn = AsymmetricLoss(
                gamma_pos=self.config.asymmetric_gamma_pos,
                gamma_neg=self.config.asymmetric_gamma_neg
            )
            logger.info(f"Using asymmetric loss with gamma_pos={self.config.asymmetric_gamma_pos}, "
                       f"gamma_neg={self.config.asymmetric_gamma_neg}")

        elif self.config.loss_type == "smoothing":
            self.loss_fn = LabelSmoothingLoss(
                smoothing=self.config.smoothing_factor
            )
            logger.info(f"Using label smoothing loss with smoothing={self.config.smoothing_factor}")

        else:
            # Default: Binary Cross-Entropy Loss
            self.loss_fn = None
            logger.info("Using default BCE loss")

    def _create_dataloaders(self, data_splits: Dict[str, Dict[str, Any]]) -> Dict[str, DataLoader]:
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
                tokenizer_name=self.config.base_model,
                max_length=self.config.max_length
            )

            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == "train"),
                collate_fn=collator,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=(self.config.device != "cpu")
            )

            # Apply memory optimizations
            dataloader = self.memory_optimizer.optimize_dataloader(dataloader)

            # Set up distributed training if enabled
            if split == "train" and self.distributed_manager.setup():
                dataloader = self.distributed_manager.prepare_dataloader(dataloader)

            dataloaders[split] = dataloader

        return dataloaders

    def _create_optimizer_and_scheduler(self, dataloader: DataLoader) -> Tuple[Optimizer, Any]:
        """
        Create optimizer and learning rate scheduler.

        Args:
            dataloader: Training dataloader

        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Prepare optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate
        )

        # Calculate total steps and warmup steps
        total_steps = len(dataloader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_steps_fraction)

        # Create scheduler
        if self.config.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            logger.info(f"Using cosine scheduler with {warmup_steps} warmup steps")
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            logger.info(f"Using linear scheduler with {warmup_steps} warmup steps")

        return optimizer, scheduler

    def train(self, data_splits: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the topic classification model with advanced techniques.

        Args:
            data_splits: Dictionary containing train/val/test splits

        Returns:
            Dictionary with training results and metrics
        """
        # Create dataloaders
        dataloaders = self._create_dataloaders(data_splits)

        # Create optimizer and scheduler
        optimizer, scheduler = self._create_optimizer_and_scheduler(dataloaders["train"])

        # Training loop
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        start_time = time.time()

        # Load checkpoint if available
        start_epoch, start_step, _ = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler
        )

        # Training loop
        for epoch in range(start_epoch, self.config.num_epochs):
            # Training
            train_metrics = self._train_epoch(
                dataloader=dataloaders["train"],
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch
            )

            # Validation
            val_metrics = self._evaluate(dataloaders["val"])

            # Log progress
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs} - "
                       f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val F1: {val_metrics['f1']:.4f}")

            # Check if this is the best model
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict() if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
                                       else self.model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics["loss"],
                    'val_f1': val_metrics["f1"],
                }
                logger.info(f"New best model saved (F1: {val_metrics['f1']:.4f})")

            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=train_metrics["steps"],
                metrics=val_metrics,
                is_best=is_best
            )

            # Store stats
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': train_metrics["loss"],
                'val_loss': val_metrics["loss"],
                'val_precision': val_metrics["precision"],
                'val_recall': val_metrics["recall"],
                'val_f1': val_metrics["f1"],
                'val_auc': val_metrics.get("auc", 0.0),
                'learning_rate': train_metrics["learning_rate"]
            }
            self.training_stats.append(epoch_stats)

            # Check early stopping
            if self.checkpoint_manager.check_early_stopping(val_metrics["loss"]):
                logger.info("Early stopping triggered")
                break

        # Training completed
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        # Load best model
        if self.best_model_state:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(self.best_model_state["model_state_dict"])
            else:
                self.model.load_state_dict(self.best_model_state["model_state_dict"])
            logger.info(f"Loaded best model from epoch {self.best_model_state['epoch']}")

        # Test on held-out test set
        test_metrics = self._evaluate(dataloaders["test"])
        logger.info(f"Test metrics - "
                   f"Loss: {test_metrics['loss']:.4f}, "
                   f"F1: {test_metrics['f1']:.4f}, "
                   f"Precision: {test_metrics['precision']:.4f}, "
                   f"Recall: {test_metrics['recall']:.4f}, "
                   f"AUC: {test_metrics.get('auc', 0.0):.4f}")

        # Clean up distributed training
        self.distributed_manager.cleanup()

        return {
            "training_stats": self.training_stats,
            "best_model_state": self.best_model_state,
            "test_metrics": test_metrics,
            "training_time": training_time
        }

    def _train_epoch(self, dataloader: DataLoader, optimizer: Optimizer,
                    scheduler: Any, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_steps = 0

        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for step_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.config.device) for k, v in batch.items()}

            # Apply mixup if enabled
            if self.mixup is not None:
                batch, _ = self.mixup(batch)

            # Apply adversarial training if enabled
            if self.adversarial is not None and step_idx % 5 == 0:  # Apply every 5 steps to save computation
                adv_batch, _, _ = self.adversarial.generate_adversarial_examples(batch)
                # Train on both original and adversarial examples
                batches = [batch, adv_batch]
            else:
                batches = [batch]

            # Process all batches (original and adversarial if applicable)
            for b in batches:
                # Forward pass with mixed precision
                loss, outputs = self.mixed_precision.forward_backward(
                    model=self.model,
                    batch=b,
                    loss_fn=self.loss_fn
                )

                # Backward pass with gradient accumulation
                self.mixed_precision.backward_step(
                    loss=loss,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_accum_steps=self.config.optimization.gradient_accumulation_steps,
                    step_idx=step_idx,
                    max_grad_norm=self.config.optimization.max_grad_norm
                )

                # Update metrics
                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

            total_steps += 1

        # Calculate average metrics
        avg_loss = total_loss / total_steps

        return {
            "loss": avg_loss,
            "steps": total_steps,
            "learning_rate": scheduler.get_last_lr()[0]
        }

    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
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
                batch = {k: v.to(self.config.device) for k, v in batch.items()}

                # Forward pass
                if self.loss_fn is None:
                    # Use model's built-in loss calculation
                    outputs = self.model(**batch)
                    loss = outputs.loss
                else:
                    # Use custom loss function
                    outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                    loss = self.loss_fn(outputs.logits, batch["labels"])

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

        # Calculate AUC if possible (requires sklearn 0.24+)
        auc = roc_auc_score(all_labels, all_preds, average='micro')


        avg_loss = total_loss / len(dataloader)

        return {
            "loss": avg_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    def save_model(self, path: str):
        """
        Save the trained model.

        Args:
            path: Directory to save the model
        """
        os.makedirs(path, exist_ok=True)

        # Get model to save (unwrap DDP if needed)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        # Save model state
        model_path = os.path.join(path, "model.pt")
        torch.save(model_to_save.state_dict(), model_path)

        # Save configuration
        model_to_save.config.save_pretrained(path)

        # Save training configuration
        config_path = os.path.join(path, "training_config.json")
        with open(config_path, "w") as f:
            import json
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load a trained model.

        Args:
            path: Path to the saved model
        """
        # Get model to load (unwrap DDP if needed)
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_to_load = self.model.module
        else:
            model_to_load = self.model

        # Load model state
        model_path = os.path.join(path, "model.pt")
        model_to_load.load_state_dict(torch.load(model_path, map_location=self.config.device))

        logger.info(f"Model loaded from {path}")
