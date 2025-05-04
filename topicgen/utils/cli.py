"""
Common utilities for command-line interfaces, argument parsing, and logging.
"""

import argparse
import logging
from typing import Optional

# Configure default logging
def setup_logging(level: int = logging.INFO, format_str: Optional[str] = None) -> logging.Logger:
    """
    Set up logging with consistent format across the application.

    Args:
        level: Logging level (default: INFO)
        format_str: Custom format string (optional)

    Returns:
        Logger instance
    """
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=level,
        format=format_str
    )

    return logging.getLogger(__name__)

# Common argument parsers
def get_data_collection_parser() -> argparse.ArgumentParser:
    """
    Get argument parser for data collection pipeline.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="GitHub Repository Data Collection Pipeline")

    # Data collection parameters
    parser.add_argument("--min-stars", type=int, default=1000,
                       help="Minimum stars for repositories")
    parser.add_argument("--max-stars", type=int, default=50000,
                       help="Maximum stars for repositories")
    parser.add_argument("--language", type=str, default="python",
                       help="Programming language to filter")
    parser.add_argument("--max-repos", type=int, default=1000,
                       help="Maximum repositories to collect")
    parser.add_argument("--incremental", action="store_true",
                       help="Use incremental collection")
    parser.add_argument("--update-days", type=int, default=7,
                       help="Days since last update to refresh")
    parser.add_argument("--cache-ttl", type=int, default=3600,
                       help="Cache TTL in seconds")
    parser.add_argument("--use-mock-api", action="store_true",
                       help="Use mock GitHub API instead of real API")

    return parser

def get_model_training_parser() -> argparse.ArgumentParser:
    """
    Get argument parser for model training pipeline.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="GitHub Topic Model Training Pipeline")

    # Model configuration
    parser.add_argument("--base-model", type=str, default="bert-base-uncased",
                       help="Base transformer model")
    parser.add_argument("--min-topic-count", type=int, default=10,
                       help="Minimum topic occurrences to include")
    parser.add_argument("--max-topics", type=int, default=500,
                       help="Maximum number of topics to classify")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay for regularization")
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length for tokenization")

    # Scheduler parameters
    parser.add_argument("--scheduler-type", type=str, default="linear", choices=["linear", "cosine"],
                       help="Type of learning rate scheduler")
    parser.add_argument("--warmup-steps-fraction", type=float, default=0.1,
                       help="Fraction of steps for warmup")

    # Loss function
    parser.add_argument("--loss-type", type=str, default="bce",
                       choices=["bce", "focal", "asymmetric", "smoothing"],
                       help="Type of loss function")

    # Advanced techniques
    parser.add_argument("--use-mixup", action="store_true",
                       help="Use mixup data augmentation")
    parser.add_argument("--use-adversarial", action="store_true",
                       help="Use adversarial training")

    # Optimization techniques
    parser.add_argument("--use-mixed-precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Number of steps to accumulate gradients")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                       help="Number of epochs to wait for improvement before stopping")

    # Data and output
    parser.add_argument("--data-limit", type=int, default=10000,
                       help="Maximum training examples to use")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save model")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for training (cuda, cpu, or None for auto-detection)")

    # Hyperparameter tuning
    parser.add_argument("--tune-hyperparameters", action="store_true",
                       help="Perform hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=5,
                       help="Number of hyperparameter combinations to try")

    # Feature selection
    parser.add_argument("--basic-mode", action="store_true",
                       help="Use basic training features only (disable enhanced features)")

    return parser

def get_pipeline_parser() -> argparse.ArgumentParser:
    """
    Get argument parser for the unified pipeline.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="TopicGen Pipeline Runner")

    # Pipeline selection
    parser.add_argument("--skip-data-collection", action="store_true",
                       help="Skip data collection phase")
    parser.add_argument("--skip-model-training", action="store_true",
                       help="Skip model training phase")

    # Add data collection arguments
    data_parser = get_data_collection_parser()
    for action in data_parser._actions:
        if action.dest != 'help':
            parser._add_action(action)

    # Add model training arguments
    model_parser = get_model_training_parser()
    for action in model_parser._actions:
        if action.dest != 'help' and not any(a.dest == action.dest for a in parser._actions):
            parser._add_action(action)

    return parser

def get_test_data_collection_parser() -> argparse.ArgumentParser:
    """
    Get argument parser for testing the data collection pipeline.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(description="GitHub Repository Data Collection Test")
    parser.add_argument("--use-real-api", action="store_true",
                       help="Use real GitHub API instead of mock")
    parser.add_argument("--min-stars", type=int, default=1000,
                       help="Minimum stars for repositories")
    parser.add_argument("--max-stars", type=int, default=50000,
                       help="Maximum stars for repositories")
    parser.add_argument("--language", type=str, default="python",
                       help="Programming language to filter")
    parser.add_argument("--max-repos", type=int, default=10,
                       help="Maximum repositories to collect")
    parser.add_argument("--incremental", action="store_true",
                       help="Use incremental collection")
    parser.add_argument("--update-days", type=int, default=7,
                       help="Days since last update to refresh")
    parser.add_argument("--cache-ttl", type=int, default=3600,
                       help="Cache TTL in seconds")
    parser.add_argument("--show-training-data", action="store_true",
                       help="Display sample training data")
    parser.add_argument("--training-data-limit", type=int, default=5,
                       help="Number of training examples to display")

    return parser
