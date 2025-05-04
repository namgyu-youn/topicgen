import logging
from collections import Counter
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocesses repository data for model training."""

    def __init__(self, min_topic_count: int = 5, max_topics: int = 1000):
        """
        Initialize the data preprocessor.

        Args:
            min_topic_count: Minimum number of occurrences for a topic to be included
            max_topics: Maximum number of topics to consider in the model
        """
        self.min_topic_count = min_topic_count
        self.max_topics = max_topics
        self.topic_to_id = {}  # Mapping from topic string to numeric ID
        self.id_to_topic = {}  # Reverse mapping

    def preprocess_data(self, training_data: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Preprocess repository data for model training.

        Args:
            training_data: List of dictionaries containing repository content and topics

        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info(f"Preprocessing {len(training_data)} training examples")

        # Extract texts and topics
        texts = [item["content"] for item in training_data]
        all_topics_lists = [item["labels"].split(", ") for item in training_data]

        # Count topic frequencies
        topic_counts = Counter()
        for topics in all_topics_lists:
            topic_counts.update(topics)

        logger.info(f"Found {len(topic_counts)} unique topics in dataset")

        # Filter topics by minimum count
        valid_topics = [
            topic for topic, count in topic_counts.items()
            if count >= self.min_topic_count
        ]
        valid_topics = valid_topics[:self.max_topics]  # Limit to max topics

        logger.info(f"Using {len(valid_topics)} topics after filtering")

        # Create topic to ID mapping
        self.topic_to_id = {topic: idx for idx, topic in enumerate(valid_topics)}
        self.id_to_topic = {idx: topic for topic, idx in self.topic_to_id.items()}

        # Convert topic lists to multi-hot encoded vectors
        labels = []
        for topics in all_topics_lists:
            # Create multi-hot encoded vector
            label_vector = np.zeros(len(valid_topics), dtype=np.float32)
            for topic in topics:
                if topic in self.topic_to_id:
                    label_vector[self.topic_to_id[topic]] = 1.0
            labels.append(label_vector)

        logger.info(f"Converted topics to {len(labels)} multi-hot encoded vectors")

        # Return processed data
        return {
            "texts": texts,
            "labels": labels,
            "topic_to_id": self.topic_to_id,
            "id_to_topic": self.id_to_topic,
            "num_topics": len(valid_topics)
        }

    def create_train_val_test_split(self,
                                   texts: list[str],
                                   labels: list[np.ndarray],
                                   train_size: float = 0.8,
                                   val_size: float = 0.1,
                                   test_size: float = 0.1,
                                   random_state: int = 42) -> dict[str, Any]:
        """
        Split data into training, validation, and test sets.

        Args:
            texts: List of text content
            labels: List of multi-hot encoded label vectors
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing the data splits
        """
        if abs(train_size + val_size + test_size - 1.0) >= 1e-6:
            raise ValueError("Split proportions must sum to 1")

        # Convert to numpy arrays for easier handling
        labels_array = np.array(labels)

        # First split into train and temp (val+test)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels_array,
            train_size=train_size,
            random_state=random_state,
            stratify=None  # Cannot stratify with multi-label
        )

        # Then split temp into val and test
        relative_val_size = val_size / (val_size + test_size)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels,
            train_size=relative_val_size,
            random_state=random_state,
            stratify=None
        )

        logger.info(f"Data split: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

        return {
            "train": {"texts": train_texts, "labels": train_labels},
            "val": {"texts": val_texts, "labels": val_labels},
            "test": {"texts": test_texts, "labels": test_labels}
        }
