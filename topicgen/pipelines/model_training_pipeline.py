import asyncio
import json
import logging
import os
import sys
import time
from topicgen.database import DataStore
from topicgen.models import DataPreprocessor, ModelExporter, TopicTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Define constants (hardcoded values)
BASE_MODEL = "bert-base-uncased"
MIN_TOPIC_COUNT,MAX_TOPICS = 10, 500
BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS = 16, 3e-5, 40
DATA_LIMIT, OUTPUT_DIR = 10000, "models"

class ModelTrainingPipeline:
    """
    GitHub Topic Model Training Pipeline Class
    """

    def __init__(
        self,
        base_model=BASE_MODEL,
        min_topic_count=MIN_TOPIC_COUNT,
        max_topics=MAX_TOPICS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        data_limit=DATA_LIMIT,
        output_dir=OUTPUT_DIR,
        device=None
    ):
        """Initialize the pipeline"""
        self.base_model = base_model
        self.min_topic_count = min_topic_count
        self.max_topics = max_topics
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.data_limit = data_limit
        self.output_dir = output_dir
        self.device = device

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data store
        self.data_store = DataStore()

    async def fetch_training_data(self):
        """Fetch training data from database"""
        logger.info(f"Fetching training data (limit: {self.data_limit})")
        training_data = await self.data_store.get_training_data(limit=self.data_limit)

        if not training_data:
            raise ValueError("No training data retrieved from database")

        logger.info(f"Retrieved {len(training_data)} training examples")
        return training_data

    def preprocess_data(self, training_data):
        """Preprocess and split data"""
        logger.info("Preprocessing training data")
        preprocessor = DataPreprocessor(
            min_topic_count=self.min_topic_count,
            max_topics=self.max_topics
        )
        processed_data = preprocessor.preprocess_data(training_data)

        logger.info("Splitting data into train/val/test sets")
        data_splits = preprocessor.create_train_val_test_split(
            texts=processed_data["texts"],
            labels=processed_data["labels"]
        )

        return processed_data, data_splits, preprocessor

    def train_model(self, processed_data, data_splits):
        """Initialize and train model"""
        logger.info(f"Training model with {len(data_splits['train']['texts'])} examples")
        trainer = TopicTrainer(
            base_model=self.base_model,
            num_topics=processed_data["num_topics"],
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            num_epochs=self.num_epochs,
            device=self.device
        )

        logger.info(f"Using device: {trainer.device} for training")
        results = trainer.train(data_splits)

        return trainer, results

    def save_model_artifacts(self, trainer, processed_data, results):
        """Save model artifacts"""
        logger.info("Saving model artifacts")
        model_path = os.path.join(self.output_dir, "pytorch_model")
        trainer.save_model(model_path)

        # Save topic mappings
        mapping_path = os.path.join(self.output_dir, "topic_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump({
                "topic_to_id": processed_data["topic_to_id"],
                "id_to_topic": {str(k): v for k, v in processed_data["id_to_topic"].items()},
                "num_topics": processed_data["num_topics"]
            }, f, indent=2)

        logger.info(f"Saved topic mapping to {mapping_path}")

        # Export model
        logger.info("Exporting model for production")
        exporter = ModelExporter(
            model=trainer.model,
            tokenizer_name=self.base_model,
            output_path=model_path,
            device=trainer.device
        )

        export_path = exporter.export_model()

        # Validate model
        validation_result = exporter.validate_model()
        validation_status = "successful" if validation_result else "failed"
        logger.info(f"Model validation {validation_status}")

        return export_path, validation_result

    async def run(self):
        """Run the complete pipeline"""
        start_time = time.time()

        try:
            logger.info(f"Starting model training pipeline with {self.base_model}")

            # 1. Fetch training data
            training_data = await self.fetch_training_data()

            # 2. Preprocess data
            processed_data, data_splits, preprocessor = self.preprocess_data(training_data)

            # 3. Train model
            trainer, results = self.train_model(processed_data, data_splits)

            # 4. Save model artifacts
            model_path, validation_result = self.save_model_artifacts(
                trainer, processed_data, results
            )

            # Calculate execution time
            execution_time = round(time.time() - start_time, 2)

            # Return success results
            return {
                "status": "success",
                "model_path": model_path,
                "num_topics": processed_data["num_topics"],
                "test_metrics": results["test_metrics"],
                "execution_time": execution_time,
                "validation": validation_result
            }

        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": round(time.time() - start_time, 2)
            }

async def run_model_training(**kwargs):
    """
    Run model training pipeline.

    Args:
        **kwargs: Parameters to pass to ModelTrainingPipeline constructor

    Returns:
        Dictionary containing training results
    """
    pipeline = ModelTrainingPipeline(**kwargs)
    return await pipeline.run()

def main():
    """Command line entry point for model training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="GitHub Topic Model Training Pipeline")
    parser.add_argument("--base-model", type=str, default=BASE_MODEL,
                       help=f"Base transformer model (default: {BASE_MODEL})")
    parser.add_argument("--min-topic-count", type=int, default=MIN_TOPIC_COUNT,
                       help=f"Minimum topic occurrences to include (default: {MIN_TOPIC_COUNT})")
    parser.add_argument("--max-topics", type=int, default=MAX_TOPICS,
                       help=f"Maximum number of topics to classify (default: {MAX_TOPICS})")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                       help=f"Directory to save model (default: {OUTPUT_DIR})")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for training (cuda, cpu, or auto-detect)")

    args = parser.parse_args()
    kwargs = vars(args)

    # Run the pipeline
    try:
        result = asyncio.run(run_model_training(**kwargs))

        if result["status"] == "success":
            print("\n===== Model Training Results =====")
            print(f"Model saved to: {result['model_path']}")
            print(f"Number of topics: {result['num_topics']}")
            print(f"Execution time: {result['execution_time']} seconds")
            print(f"Model validation: {'successful' if result['validation'] else 'failed'}")

            print("\nTest metrics:")
            for metric, value in result["test_metrics"].items():
                print(f"  - {metric}: {value:.4f}")

            return 0
        else:
            print(f"\nPipeline failed: {result.get('error', 'Unknown error')}")
            print(f"Execution time: {result.get('execution_time', 'N/A')} seconds")
            return 1

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
