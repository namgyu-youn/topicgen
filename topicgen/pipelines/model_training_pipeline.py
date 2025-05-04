"""
Unified model training pipeline with advanced techniques and optimizations.
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Optional

import torch

from topicgen.database import DataStore
from topicgen.models import DataPreprocessor
from topicgen.models.enhanced_trainer import EnhancedTopicTrainer, EnhancedTrainerConfig
from topicgen.models.training_optimizations import OptimizationConfig
from topicgen.models.hyperparameter_tuning import (
    HyperparameterSpace,
    HyperparameterTuner,
    HyperparameterConfig,
    create_subset_for_tuning
)
from topicgen.models.dataset import TopicDataset
from topicgen.utils.cli import setup_logging, get_model_training_parser

logger = setup_logging()


async def run_training(
    base_model: str = "bert-base-uncased",
    min_topic_count: int = 10,
    max_topics: int = 500,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 5,
    max_length: int = 512,
    scheduler_type: str = "linear",
    warmup_steps_fraction: float = 0.1,
    loss_type: str = "bce",
    use_mixup: bool = False,
    use_adversarial: bool = False,
    use_mixed_precision: bool = True,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 3,
    data_limit: int = 10000,
    output_dir: str = "models",
    device: Optional[str] = None,
    tune_hyperparameters: bool = False,
    n_trials: int = 5,
    use_enhanced_features: bool = True
) -> dict[str, Any]:
    """
    Run the model training pipeline with optional advanced features.

    Args:
        base_model: Base transformer model
        min_topic_count: Minimum topic occurrences to include in training
        max_topics: Maximum number of topics to classify
        batch_size: Training batch size
        learning_rate: Learning rate for training
        weight_decay: Weight decay for regularization
        num_epochs: Number of training epochs
        max_length: Maximum sequence length for tokenization
        scheduler_type: Type of learning rate scheduler ('linear' or 'cosine')
        warmup_steps_fraction: Fraction of steps for warmup
        loss_type: Type of loss function ('bce', 'focal', 'asymmetric', or 'smoothing')
        use_mixup: Whether to use mixup data augmentation
        use_adversarial: Whether to use adversarial training
        use_mixed_precision: Whether to use mixed precision training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        early_stopping_patience: Number of epochs to wait for improvement before stopping
        data_limit: Maximum number of training examples to use
        output_dir: Directory to save the model
        device: Device to use for training ('cuda', 'cpu', or None for auto-detection)
        tune_hyperparameters: Whether to perform hyperparameter tuning
        n_trials: Number of hyperparameter combinations to try
        use_enhanced_features: Whether to use enhanced training features

    Returns:
        Dictionary with training results
    """
    try:
        start_time = time.time()
        logger.info(f"Starting model training pipeline with {base_model}")
        logger.info(f"Using {'enhanced' if use_enhanced_features else 'standard'} training features")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # 1. Fetch training data from database
        logger.info(f"Fetching training data (limit: {data_limit})")
        data_store = DataStore()
        training_data = await data_store.get_training_data(limit=data_limit)

        if not training_data:
            logger.error("No training data retrieved from database")
            return {"status": "failed", "error": "No training data available"}

        logger.info(f"Retrieved {len(training_data)} training examples")

        # 2. Preprocess data
        logger.info("Preprocessing training data")
        preprocessor = DataPreprocessor(
            min_topic_count=min_topic_count,
            max_topics=max_topics
        )
        processed_data = preprocessor.preprocess_data(training_data)

        # 3. Split data into train/val/test sets
        logger.info("Splitting data into train/val/test sets")
        data_splits = preprocessor.create_train_val_test_split(
            texts=processed_data["texts"],
            labels=processed_data["labels"]
        )

        # 4. Configure training
        optimization_config = OptimizationConfig(
            use_mixed_precision=use_mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            early_stopping_patience=early_stopping_patience,
            checkpoint_interval=1,
            checkpoint_dir=os.path.join(output_dir, "checkpoints")
        )

        trainer_config = EnhancedTrainerConfig(
            base_model=base_model,
            num_topics=processed_data["num_topics"],
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            max_length=max_length,
            scheduler_type=scheduler_type,
            warmup_steps_fraction=warmup_steps_fraction,
            loss_type=loss_type,
            use_mixup=use_mixup and use_enhanced_features,
            use_adversarial=use_adversarial and use_enhanced_features,
            optimization=optimization_config if use_enhanced_features else None,
            device=device
        )

        # 5. Hyperparameter tuning if requested
        if tune_hyperparameters and use_enhanced_features:
            logger.info(f"Starting hyperparameter tuning with {n_trials} trials")

            # Define hyperparameter space
            param_space = HyperparameterSpace(
                base_models=[base_model],
                learning_rates=[1e-5, 3e-5, 5e-5],
                batch_sizes=[8, 16, 32],
                weight_decays=[0.0, 0.01, 0.1],
                warmup_steps_fractions=[0.0, 0.1, 0.2],
                scheduler_types=["linear", "cosine"],
                dropout_rates=[0.1, 0.2, 0.3],
                max_lengths=[128, 256, 512]
            )

            # Create a smaller dataset for tuning
            train_dataset = TopicDataset(
                texts=data_splits["train"]["texts"],
                labels=data_splits["train"]["labels"],
                tokenizer_name=base_model,
                max_length=max_length
            )

            train_subset = create_subset_for_tuning(train_dataset, fraction=0.3, max_examples=2000)

            # Define training function for tuner
            def train_with_config(config: HyperparameterConfig) -> dict[str, float]:
                # Convert to EnhancedTrainerConfig
                tuning_config = EnhancedTrainerConfig(
                    base_model=config.base_model,
                    num_topics=processed_data["num_topics"],
                    dropout_rate=config.dropout_rate,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    num_epochs=3,  # Use fewer epochs for tuning
                    max_length=config.max_length,
                    scheduler_type=config.scheduler_type,
                    warmup_steps_fraction=config.warmup_steps_fraction,
                    loss_type=loss_type,
                    use_mixup=use_mixup,
                    use_adversarial=use_adversarial,
                    optimization=OptimizationConfig(
                        use_mixed_precision=use_mixed_precision,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        early_stopping_patience=0  # Disable early stopping for tuning
                    ),
                    device=device
                )

                # Create trainer
                trainer = EnhancedTopicTrainer(config=tuning_config)

                # Create smaller data splits for tuning
                tuning_splits = {
                    "train": {
                        "texts": [train_subset.dataset.texts[i] for i in train_subset.indices],
                        "labels": train_subset.dataset.labels[train_subset.indices]
                    },
                    "val": {
                        "texts": data_splits["val"]["texts"],
                        "labels": data_splits["val"]["labels"]
                    },
                    "test": {
                        "texts": data_splits["val"]["texts"],  # Use val as test for tuning
                        "labels": data_splits["val"]["labels"]
                    }
                }

                # Train and evaluate
                results = trainer.train(tuning_splits)

                # Return validation metrics
                return {
                    "val_loss": results["best_model_state"]["val_loss"],
                    "val_f1": results["best_model_state"]["val_f1"]
                }

            # Run hyperparameter tuning
            tuner = HyperparameterTuner(
                param_space=param_space,
                train_fn=train_with_config,
                metric_name="val_f1",
                metric_mode="max",
                n_trials=n_trials,
                search_strategy="random",
                output_dir=os.path.join(output_dir, "tuning"),
                use_gpu=(device == "cuda" or (device is None and torch.cuda.is_available()))
            )

            best_config, best_results = tuner.run()

            # Update trainer config with best hyperparameters
            trainer_config.base_model = best_config.base_model
            trainer_config.learning_rate = best_config.learning_rate
            trainer_config.batch_size = best_config.batch_size
            trainer_config.weight_decay = best_config.weight_decay
            trainer_config.scheduler_type = best_config.scheduler_type
            trainer_config.warmup_steps_fraction = best_config.warmup_steps_fraction
            trainer_config.dropout_rate = best_config.dropout_rate
            trainer_config.max_length = best_config.max_length

            logger.info(f"Hyperparameter tuning completed. Best F1: {best_results['metrics']['val_f1']:.4f}")
            logger.info(f"Best hyperparameters: {best_config}")

            # Save best hyperparameters
            with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
                json.dump(best_config.to_dict(), f, indent=2)

        # 6. Train model with final configuration
        logger.info(f"Training model with {len(data_splits['train']['texts'])} examples")
        logger.info(f"Using configuration: {trainer_config}")

        trainer = EnhancedTopicTrainer(config=trainer_config)

        # Log device information
        logger.info(f"Using device: {trainer_config.device} for training")

        results = trainer.train(data_splits)

        # 7. Save model artifacts
        logger.info("Saving model artifacts")
        model_path = os.path.join(output_dir, "pytorch_model")
        trainer.save_model(model_path)

        # Save topic mappings
        with open(os.path.join(output_dir, "topic_mapping.json"), "w") as f:
            json.dump({
                "topic_to_id": processed_data["topic_to_id"],
                "id_to_topic": {str(k): v for k, v in processed_data["id_to_topic"].items()},
                "num_topics": processed_data["num_topics"]
            }, f, indent=2)

        # Save training stats
        with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
            json.dump(results["training_stats"], f, indent=2)

        # Calculate total time
        total_time = time.time() - start_time

        logger.info(f"Model training pipeline completed successfully in {total_time:.2f} seconds")

        return {
            "status": "success",
            "model_path": model_path,
            "num_topics": processed_data["num_topics"],
            "test_metrics": results["test_metrics"],
            "training_time": total_time
        }

    except Exception as e:
        logger.error(f"Model training pipeline failed: {e!s}", exc_info=True)
        return {"status": "failed", "error": str(e)}


def main():
    """Command line entry point for model training pipeline."""
    parser = get_model_training_parser()

    args = parser.parse_args()

    # Run the pipeline
    try:
        result = asyncio.run(run_training(
            base_model=args.base_model,
            min_topic_count=args.min_topic_count,
            max_topics=args.max_topics,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            scheduler_type=args.scheduler_type,
            warmup_steps_fraction=args.warmup_steps_fraction,
            loss_type=args.loss_type,
            use_mixup=args.use_mixup,
            use_adversarial=args.use_adversarial,
            use_mixed_precision=args.use_mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            early_stopping_patience=args.early_stopping_patience,
            data_limit=args.data_limit,
            output_dir=args.output_dir,
            device=args.device,
            tune_hyperparameters=args.tune_hyperparameters,
            n_trials=args.n_trials,
            use_enhanced_features=not args.basic_mode
        ))

        if result and result["status"] == "success":
            print("\n===== Model Training Results =====")
            print(f"Model saved to: {result['model_path']}")
            print(f"Number of topics: {result['num_topics']}")
            print(f"Training time: {result['training_time']:.2f} seconds")
            print("Test metrics:")
            for metric, value in result["test_metrics"].items():
                print(f"  - {metric}: {value:.4f}")
        else:
            print("Pipeline failed. Check logs for details.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
