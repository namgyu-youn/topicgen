"""
Training optimization techniques for improving model performance and efficiency.
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Tuple

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for training optimizations."""

    # Mixed precision training
    use_mixed_precision: bool = False

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Distributed training
    use_distributed: bool = False
    local_rank: int = -1
    world_size: int = 1

    # Memory optimizations
    optimize_memory_usage: bool = False

    # Checkpointing
    checkpoint_interval: int = 0  # 0 means no checkpointing during training
    checkpoint_dir: str = "checkpoints"

    # Early stopping
    early_stopping_patience: int = 0  # 0 means no early stopping
    early_stopping_threshold: float = 0.001

    # Gradient clipping
    max_grad_norm: float = 1.0


class MixedPrecisionTrainer:
    """Implements mixed precision training for improved performance on GPUs."""

    def __init__(self, enabled: bool = True):
        """
        Initialize mixed precision trainer.

        Args:
            enabled: Whether to enable mixed precision training
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enabled)

        if self.enabled:
            logger.info("Mixed precision training enabled")
        else:
            if enabled and not torch.cuda.is_available():
                logger.warning("Mixed precision requested but CUDA not available, disabled")
            else:
                logger.info("Mixed precision training disabled")

    def forward_backward(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor],
                        loss_fn: Callable = None) -> torch.Tensor:
        """
        Perform forward and backward pass with mixed precision.

        Args:
            model: PyTorch model
            batch: Input batch
            loss_fn: Optional custom loss function, if None uses model's built-in loss

        Returns:
            Loss value
        """
        # Forward pass with autocast for mixed precision
        with autocast(enabled=self.enabled):
            if loss_fn is None:
                # Use model's built-in loss calculation
                outputs = model(**batch)
                loss = outputs.loss
            else:
                # Use custom loss function
                outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = loss_fn(outputs.logits, batch["labels"])

        return loss, outputs

    def backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                     scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                     grad_accum_steps: int = 1, step_idx: int = 0,
                     max_grad_norm: float = 1.0):
        """
        Perform backward pass with gradient scaling.

        Args:
            loss: Loss tensor
            optimizer: PyTorch optimizer
            scheduler: Optional learning rate scheduler
            grad_accum_steps: Number of gradient accumulation steps
            step_idx: Current step index within the accumulation cycle
            max_grad_norm: Maximum gradient norm for clipping
        """
        # Scale loss for gradient accumulation
        scaled_loss = loss / grad_accum_steps

        # Backward pass with gradient scaling
        self.scaler.scale(scaled_loss).backward()

        # Only update weights after accumulating enough gradients
        if (step_idx + 1) % grad_accum_steps == 0:
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                          max_norm=max_grad_norm)

            # Update weights with scaled gradients
            self.scaler.step(optimizer)

            # Update scaler for next iteration
            self.scaler.update()

            # Zero gradients
            optimizer.zero_grad()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()


class DistributedTrainingManager:
    """Manages distributed training across multiple GPUs."""

    def __init__(self, config: OptimizationConfig):
        """
        Initialize distributed training manager.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.initialized = False

        if not self.config.use_distributed:
            logger.info("Distributed training disabled")
            return

        if not torch.cuda.is_available():
            logger.warning("Distributed training requested but CUDA not available, disabled")
            self.config.use_distributed = False
            return

        if torch.cuda.device_count() < 2 and self.config.world_size <= 1:
            logger.warning("Distributed training requested but only one GPU available, disabled")
            self.config.use_distributed = False
            return

        logger.info(f"Initializing distributed training with world_size={self.config.world_size}")

    def setup(self) -> bool:
        """
        Set up distributed training environment.

        Returns:
            True if setup successful, False otherwise
        """
        if not self.config.use_distributed or self.initialized:
            return self.config.use_distributed

        try:
            # Initialize process group
            if self.config.local_rank == -1:
                logger.warning("local_rank not set, using RANK environment variable")
                self.config.local_rank = int(os.environ.get("RANK", "0"))

            # Set device
            torch.cuda.set_device(self.config.local_rank)

            # Initialize distributed process group
            dist.init_process_group(backend="nccl",
                                   world_size=self.config.world_size,
                                   rank=self.config.local_rank)

            self.initialized = True
            logger.info(f"Distributed training initialized on rank {self.config.local_rank}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e!s}")
            self.config.use_distributed = False
            return False

    def prepare_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Prepare model for distributed training.

        Args:
            model: PyTorch model

        Returns:
            Model wrapped for distributed training
        """
        if not self.config.use_distributed or not self.initialized:
            return model

        # Move model to correct device
        model = model.to(self.config.local_rank)

        # Wrap model with DDP
        model = DDP(model,
                   device_ids=[self.config.local_rank],
                   output_device=self.config.local_rank,
                   find_unused_parameters=False)

        logger.info(f"Model wrapped with DDP on rank {self.config.local_rank}")
        return model

    def prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Prepare dataloader for distributed training.

        Args:
            dataloader: PyTorch dataloader

        Returns:
            Dataloader configured for distributed training
        """
        if not self.config.use_distributed or not self.initialized:
            return dataloader

        # Create distributed sampler
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.config.world_size,
            rank=self.config.local_rank,
            shuffle=dataloader.shuffle
        )

        # Create new dataloader with distributed sampler
        new_dataloader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=True
        )

        logger.info(f"Dataloader configured for distributed training on rank {self.config.local_rank}")
        return new_dataloader

    def cleanup(self):
        """Clean up distributed training environment."""
        if self.config.use_distributed and self.initialized:
            dist.destroy_process_group()
            logger.info("Distributed training cleaned up")


class MemoryOptimizer:
    """Implements memory optimization techniques for training large models."""

    def __init__(self, enabled: bool = True):
        """
        Initialize memory optimizer.

        Args:
            enabled: Whether to enable memory optimizations
        """
        self.enabled = enabled

        if self.enabled:
            logger.info("Memory optimizations enabled")
        else:
            logger.info("Memory optimizations disabled")

    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply memory optimizations to model.

        Args:
            model: PyTorch model

        Returns:
            Optimized model
        """
        if not self.enabled:
            return model

        # Enable gradient checkpointing if available
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        return model

    def optimize_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Apply memory optimizations to dataloader.

        Args:
            dataloader: PyTorch dataloader

        Returns:
            Optimized dataloader
        """
        if not self.enabled:
            return dataloader

        # Create a new dataloader with optimized settings
        # Use the same configuration as the original dataloader but add pin_memory
        kwargs = {
            'dataset': dataloader.dataset,
            'batch_size': dataloader.batch_size,
            'sampler': dataloader.sampler,
            'num_workers': dataloader.num_workers,
            'collate_fn': dataloader.collate_fn,
            'pin_memory': True,
        }

        # Add prefetch_factor if num_workers > 0
        if dataloader.num_workers > 0:
            kwargs['prefetch_factor'] = 2

        # If no sampler is specified, we need to set shuffle
        # (when a sampler is provided, shuffle must be False)
        if dataloader.sampler is None:
            # Try to infer if the original dataloader was using shuffle
            # Default to False if we can't determine
            kwargs['shuffle'] = False

        # Create new dataloader with optimized settings
        new_dataloader = DataLoader(**kwargs)

        logger.info("Dataloader memory optimizations applied")
        return new_dataloader


class CheckpointManager:
    """Manages model checkpointing during training."""

    def __init__(self, config: OptimizationConfig):
        """
        Initialize checkpoint manager.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.best_metric = float('inf')  # Lower is better for loss
        self.patience_counter = 0
        self.best_checkpoint_path = None

        if self.config.checkpoint_interval > 0:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            logger.info(f"Checkpoint manager initialized with interval {self.config.checkpoint_interval}")

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int, step: int, metrics: Dict[str, float],
                       is_best: bool = False) -> str:
        """
        Save model checkpoint.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch
            step: Current step
            metrics: Evaluation metrics
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        # Skip if checkpointing is disabled
        if self.config.checkpoint_interval <= 0 and not is_best:
            return None

        # Unwrap model if using DDP
        if isinstance(model, DDP):
            model_to_save = model.module
        else:
            model_to_save = model

        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        # Save checkpoint
        if is_best:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            self.best_checkpoint_path = checkpoint_path
        else:
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

        return checkpoint_path

    def load_checkpoint(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       checkpoint_path: Optional[str] = None) -> Tuple[int, int, Dict[str, float]]:
        """
        Load model checkpoint.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            scheduler: Learning rate scheduler
            checkpoint_path: Path to checkpoint, if None loads best checkpoint

        Returns:
            Tuple of (epoch, step, metrics)
        """
        if checkpoint_path is None:
            checkpoint_path = self.best_checkpoint_path

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return 0, 0, {}

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # Unwrap model if using DDP
        if isinstance(model, DDP):
            model_to_load = model.module
        else:
            model_to_load = model

        # Load model state
        model_to_load.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

        return checkpoint.get('epoch', 0), checkpoint.get('step', 0), checkpoint.get('metrics', {})

    def check_early_stopping(self, metric_value: float) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            metric_value: Current value of the metric to monitor

        Returns:
            True if training should stop, False otherwise
        """
        # Skip if early stopping is disabled
        if self.config.early_stopping_patience <= 0:
            return False

        # Check if metric improved
        if metric_value < self.best_metric - self.config.early_stopping_threshold:
            # Metric improved
            self.best_metric = metric_value
            self.patience_counter = 0
            return False
        else:
            # Metric did not improve
            self.patience_counter += 1

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.patience_counter} iterations without improvement")
                return True

            logger.info(f"Early stopping patience: {self.patience_counter}/{self.config.early_stopping_patience}")
            return False
