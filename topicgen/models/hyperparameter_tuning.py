"""
Hyperparameter tuning utilities for optimizing model performance.
"""

import logging
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
from sklearn.model_selection import ParameterGrid, ParameterSampler

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Defines the hyperparameter search space."""

    # Model architecture
    base_models: List[str] = field(default_factory=lambda: ["bert-base-uncased"])

    # Training parameters
    learning_rates: List[float] = field(default_factory=lambda: [1e-5, 3e-5, 5e-5])
    batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32])
    weight_decays: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.1])

    # Scheduler parameters
    warmup_steps_fractions: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    scheduler_types: List[str] = field(default_factory=lambda: ["linear", "cosine"])

    # Regularization
    dropout_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

    # Data processing
    max_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])

    def to_dict(self) -> Dict[str, List[Any]]:
        """Convert to dictionary format for parameter sampling."""
        return asdict(self)

    def save(self, path: str):
        """Save hyperparameter space to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'HyperparameterSpace':
        """Load hyperparameter space from JSON file."""
        with open(path, 'r') as f:
            params = json.load(f)
        return cls(**params)


@dataclass
class HyperparameterConfig:
    """Configuration for a specific hyperparameter combination."""

    # Model architecture
    base_model: str = "bert-base-uncased"

    # Training parameters
    learning_rate: float = 3e-5
    batch_size: int = 16
    weight_decay: float = 0.01
    num_epochs: int = 5

    # Scheduler parameters
    warmup_steps_fraction: float = 0.1
    scheduler_type: str = "linear"

    # Regularization
    dropout_rate: float = 0.1

    # Data processing
    max_length: int = 512

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'HyperparameterConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            params = json.load(f)
        return cls(**params)


class HyperparameterTuner:
    """Implements hyperparameter tuning strategies."""

    def __init__(self,
                param_space: HyperparameterSpace,
                train_fn: Callable,
                metric_name: str = "val_loss",
                metric_mode: str = "min",
                n_trials: int = 10,
                search_strategy: str = "random",
                output_dir: str = "hparam_tuning",
                use_gpu: bool = True):
        """
        Initialize hyperparameter tuner.

        Args:
            param_space: Hyperparameter search space
            train_fn: Function to train and evaluate a model with given hyperparameters
            metric_name: Name of the metric to optimize
            metric_mode: 'min' or 'max' depending on whether lower or higher metric is better
            n_trials: Number of hyperparameter combinations to try
            search_strategy: 'grid' or 'random'
            output_dir: Directory to save results
            use_gpu: Whether to use GPU for training
        """
        self.param_space = param_space
        self.train_fn = train_fn
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.n_trials = n_trials
        self.search_strategy = search_strategy
        self.output_dir = output_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results tracking
        self.results = []
        self.best_config = None
        self.best_metric = float('inf') if metric_mode == "min" else float('-inf')

        logger.info(f"Initialized hyperparameter tuner with {search_strategy} search strategy")
        logger.info(f"Will run {n_trials} trials, optimizing {metric_name} ({metric_mode})")

    def _generate_configs(self) -> List[HyperparameterConfig]:
        """
        Generate hyperparameter configurations to evaluate.

        Returns:
            List of hyperparameter configurations
        """
        param_dict = self.param_space.to_dict()

        if self.search_strategy == "grid":
            # Grid search - try all combinations
            param_combinations = list(ParameterGrid(param_dict))
            # Limit to n_trials if specified
            if len(param_combinations) > self.n_trials:
                logger.warning(f"Grid search would generate {len(param_combinations)} combinations, "
                              f"limiting to {self.n_trials} random samples")
                indices = np.random.choice(len(param_combinations), self.n_trials, replace=False)
                param_combinations = [param_combinations[i] for i in indices]
        else:
            # Random search - sample random combinations
            param_combinations = list(ParameterSampler(param_dict, n_iter=self.n_trials, random_state=42))

        # Convert to HyperparameterConfig objects
        configs = [HyperparameterConfig(**params) for params in param_combinations]

        logger.info(f"Generated {len(configs)} hyperparameter configurations to evaluate")
        return configs

    def _is_better(self, current: float, best: float) -> bool:
        """
        Check if current metric is better than best metric.

        Args:
            current: Current metric value
            best: Best metric value so far

        Returns:
            True if current is better than best
        """
        if self.metric_mode == "min":
            return current < best
        else:
            return current > best

    def run(self) -> Tuple[HyperparameterConfig, Dict[str, Any]]:
        """
        Run hyperparameter tuning.

        Returns:
            Tuple of (best_config, best_results)
        """
        configs = self._generate_configs()
        start_time = time.time()

        for i, config in enumerate(configs):
            logger.info(f"Trial {i+1}/{len(configs)}: {config}")

            # Train and evaluate model with this config
            try:
                trial_start = time.time()
                results = self.train_fn(config)
                trial_duration = time.time() - trial_start

                # Extract metric
                if isinstance(results, dict) and self.metric_name in results:
                    metric_value = results[self.metric_name]
                else:
                    logger.warning(f"Metric {self.metric_name} not found in results, using default value")
                    metric_value = float('inf') if self.metric_mode == "min" else float('-inf')

                # Track results
                trial_results = {
                    "config": config.to_dict(),
                    "metrics": results,
                    "trial": i + 1,
                    "duration": trial_duration
                }
                self.results.append(trial_results)

                # Update best if better
                if self._is_better(metric_value, self.best_metric):
                    self.best_metric = metric_value
                    self.best_config = config

                    # Save best config
                    config.save(os.path.join(self.output_dir, "best_config.json"))

                    logger.info(f"New best {self.metric_name}: {metric_value:.4f}")

                # Save all results
                with open(os.path.join(self.output_dir, "tuning_results.json"), "w") as f:
                    json.dump(self.results, f, indent=2)

            except Exception as e:
                logger.error(f"Error in trial {i+1}: {e!s}")
                continue

        total_duration = time.time() - start_time
        logger.info(f"Hyperparameter tuning completed in {total_duration:.2f} seconds")
        logger.info(f"Best {self.metric_name}: {self.best_metric:.4f}")
        logger.info(f"Best config: {self.best_config}")

        # Return best configuration and results
        best_trial = next((r for r in self.results if r["config"] == self.best_config.to_dict()), None)
        return self.best_config, best_trial


class CrossValidationTuner:
    """Implements k-fold cross-validation for hyperparameter tuning."""

    def __init__(self,
                param_space: HyperparameterSpace,
                train_fn: Callable,
                dataset: torch.utils.data.Dataset,
                n_splits: int = 5,
                metric_name: str = "val_loss",
                metric_mode: str = "min",
                n_trials: int = 5,
                output_dir: str = "cv_tuning"):
        """
        Initialize cross-validation tuner.

        Args:
            param_space: Hyperparameter search space
            train_fn: Function to train and evaluate a model with given hyperparameters and fold indices
            dataset: Full dataset to split into folds
            n_splits: Number of cross-validation folds
            metric_name: Name of the metric to optimize
            metric_mode: 'min' or 'max' depending on whether lower or higher metric is better
            n_trials: Number of hyperparameter combinations to try
            output_dir: Directory to save results
        """
        self.param_space = param_space
        self.train_fn = train_fn
        self.dataset = dataset
        self.n_splits = n_splits
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.n_trials = n_trials
        self.output_dir = output_dir

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results tracking
        self.results = []
        self.best_config = None
        self.best_metric = float('inf') if metric_mode == "min" else float('-inf')

        logger.info(f"Initialized {n_splits}-fold cross-validation tuner")
        logger.info(f"Will run {n_trials} trials, optimizing {metric_name} ({metric_mode})")

    def _generate_folds(self) -> List[Tuple[List[int], List[int]]]:
        """
        Generate cross-validation folds.

        Returns:
            List of (train_indices, val_indices) tuples
        """
        # Create random indices
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)

        # Split into folds
        fold_size = len(indices) // self.n_splits
        folds = []

        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else len(indices)

            val_indices = indices[start:end].tolist()
            train_indices = [idx for idx in indices if idx not in val_indices]

            folds.append((train_indices, val_indices))

        logger.info(f"Generated {len(folds)} cross-validation folds")
        return folds

    def _is_better(self, current: float, best: float) -> bool:
        """
        Check if current metric is better than best metric.

        Args:
            current: Current metric value
            best: Best metric value so far

        Returns:
            True if current is better than best
        """
        if self.metric_mode == "min":
            return current < best
        else:
            return current > best

    def run(self) -> Tuple[HyperparameterConfig, Dict[str, Any]]:
        """
        Run cross-validation hyperparameter tuning.

        Returns:
            Tuple of (best_config, best_results)
        """
        # Generate hyperparameter configurations
        tuner = HyperparameterTuner(
            param_space=self.param_space,
            train_fn=lambda x: None,  # Dummy function, not used
            metric_name=self.metric_name,
            metric_mode=self.metric_mode,
            n_trials=self.n_trials,
            search_strategy="random",
            output_dir=self.output_dir
        )
        configs = tuner._generate_configs()

        # Generate folds
        folds = self._generate_folds()

        start_time = time.time()

        for i, config in enumerate(configs):
            logger.info(f"Trial {i+1}/{len(configs)}: {config}")

            # Cross-validation for this config
            fold_metrics = []

            for fold_idx, (train_indices, val_indices) in enumerate(folds):
                logger.info(f"  Fold {fold_idx+1}/{len(folds)}")

                try:
                    # Train and evaluate on this fold
                    fold_results = self.train_fn(config, train_indices, val_indices)

                    # Extract metric
                    if isinstance(fold_results, dict) and self.metric_name in fold_results:
                        metric_value = fold_results[self.metric_name]
                        fold_metrics.append(metric_value)
                        logger.info(f"  Fold {fold_idx+1} {self.metric_name}: {metric_value:.4f}")
                    else:
                        logger.warning(f"Metric {self.metric_name} not found in results for fold {fold_idx+1}")

                except Exception as e:
                    logger.error(f"Error in fold {fold_idx+1}: {e!s}")
                    continue

            # Calculate average metric across folds
            if fold_metrics:
                avg_metric = sum(fold_metrics) / len(fold_metrics)
                std_metric = np.std(fold_metrics) if len(fold_metrics) > 1 else 0.0

                logger.info(f"Config {i+1} average {self.metric_name}: {avg_metric:.4f} ± {std_metric:.4f}")

                # Track results
                trial_results = {
                    "config": config.to_dict(),
                    f"avg_{self.metric_name}": avg_metric,
                    f"std_{self.metric_name}": std_metric,
                    "fold_metrics": fold_metrics,
                    "trial": i + 1
                }
                self.results.append(trial_results)

                # Update best if better
                if self._is_better(avg_metric, self.best_metric):
                    self.best_metric = avg_metric
                    self.best_config = config

                    # Save best config
                    config.save(os.path.join(self.output_dir, "best_config.json"))

                    logger.info(f"New best average {self.metric_name}: {avg_metric:.4f}")

                # Save all results
                with open(os.path.join(self.output_dir, "cv_results.json"), "w") as f:
                    json.dump(self.results, f, indent=2)

        total_duration = time.time() - start_time
        logger.info(f"Cross-validation tuning completed in {total_duration:.2f} seconds")
        logger.info(f"Best average {self.metric_name}: {self.best_metric:.4f}")
        logger.info(f"Best config: {self.best_config}")

        # Return best configuration and results
        best_trial = next((r for r in self.results if r["config"] == self.best_config.to_dict()), None)
        return self.best_config, best_trial


def create_subset_for_tuning(dataset: torch.utils.data.Dataset, fraction: float = 0.3,
                           min_examples: int = 1000, max_examples: int = 5000,
                           random_state: int = 42) -> torch.utils.data.Dataset:
    """
    Create a smaller subset of the dataset for faster hyperparameter tuning.

    Args:
        dataset: Full dataset
        fraction: Fraction of dataset to use
        min_examples: Minimum number of examples
        max_examples: Maximum number of examples
        random_state: Random seed for reproducibility

    Returns:
        Subset of the dataset
    """
    # Calculate subset size
    full_size = len(dataset)
    subset_size = int(full_size * fraction)
    subset_size = max(min(subset_size, max_examples), min_examples)
    subset_size = min(subset_size, full_size)

    # Generate random indices
    np.random.seed(random_state)
    indices = np.random.choice(full_size, subset_size, replace=False)

    # Create subset
    subset = Subset(dataset, indices)

    logger.info(f"Created subset with {len(subset)}/{full_size} examples ({fraction:.1%}) for tuning")
    return subset
