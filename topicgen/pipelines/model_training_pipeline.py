import argparse
import asyncio
import json
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from topicgen.database import DataStore
from topicgen.models import DataPreprocessor, ModelExporter, TopicTrainer


async def run_model_training(
    base_model: str = "bert-base-uncased",
    min_topic_count: int = 10,
    max_topics: int = 500,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    num_epochs: int = 5,
    data_limit: int = 10000,
    output_dir: str = "models"
):
    """
    Run the model training pipeline.

    Args:
        base_model: Base transformer model
        min_topic_count: Minimum topic occurrences to include in training
        max_topics: Maximum number of topics to classify
        batch_size: Training batch size
        learning_rate: Learning rate for training
        num_epochs: Number of training epochs
        data_limit: Maximum number of training examples to use
        output_dir: Directory to save the model

    Returns:
        Dictionary with training results
    """
    try:
        logger.info(f"Starting model training pipeline with {base_model}")

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

        # 4. Initialize and train model
        logger.info(f"Training model with {len(data_splits['train']['texts'])} examples")
        trainer = TopicTrainer(
            base_model=base_model,
            num_topics=processed_data["num_topics"],
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )

        results = trainer.train(data_splits)

        # 5. Save model artifacts
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

        # 6. Export to ONNX
        logger.info("Exporting model to ONNX format")
        exporter = ModelExporter(
            model=trainer.model,
            tokenizer_name=base_model,
            output_path=os.path.join(output_dir, "model.onnx")
        )

        onnx_path = exporter.export_to_onnx()

        # 7. Validate ONNX model
        validation_result = exporter.validate_onnx_model()

        if validation_result:
            logger.info("Model training pipeline completed successfully")
        else:
            logger.warning("ONNX model validation failed, check model quality")

        return {
            "status": "success",
            "model_path": model_path,
            "onnx_path": onnx_path,
            "num_topics": processed_data["num_topics"],
            "test_metrics": results["test_metrics"]
        }

    except Exception as e:
        logger.error(f"Model training pipeline failed: {e!s}", exc_info=True)
        return {"status": "failed", "error": str(e)}

def main():
    """Command line entry point for model training pipeline."""
    parser = argparse.ArgumentParser(description="GitHub Topic Model Training Pipeline")
    parser.add_argument("--base-model", type=str, default="bert-base-uncased",
                       help="Base transformer model")
    parser.add_argument("--min-topic-count", type=int, default=10,
                       help="Minimum topic occurrences to include")
    parser.add_argument("--max-topics", type=int, default=500,
                       help="Maximum number of topics to classify")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=3e-5,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--data-limit", type=int, default=10000,
                       help="Maximum training examples to use")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save model")

    args = parser.parse_args()

    # Run the pipeline
    try:
        result = asyncio.run(run_model_training(
            base_model=args.base_model,
            min_topic_count=args.min_topic_count,
            max_topics=args.max_topics,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            data_limit=args.data_limit,
            output_dir=args.output_dir
        ))

        if result and result["status"] == "success":
            print("\n===== Model Training Results =====")
            print(f"Model saved to: {result['model_path']}")
            print(f"ONNX model saved to: {result['onnx_path']}")
            print(f"Number of topics: {result['num_topics']}")
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