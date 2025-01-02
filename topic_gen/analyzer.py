from transformers import pipeline
from typing import List
from .topic_hierarchy import TOPIC_HIERARCHY

class TopicAnalyzer:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification")
        self.topic_hierarchy = TOPIC_HIERARCHY

    async def generate_topics(
        self,
        text: str,
        category: str,
        subcategory: str,
        category_threshold: float = 0.4,
        topic_threshold: float = 0.6
    ) -> List[str]:
        """
        Generate topics based on text content and selected categories.

        Args:
            text: The text to analyze
            category: Main category
            subcategory: Sub-category
            category_threshold: Minimum confidence score for category relevance
            topic_threshold: Minimum confidence score for topic generation
        """
        try:
            if category not in self.topic_hierarchy or subcategory not in self.topic_hierarchy[category]:
                return []

            category_check = self.classifier(
                text[:512],
                [category],
                multi_label=False
            )

            if category_check["scores"][0] < category_threshold:
                return []

            candidate_labels = self.topic_hierarchy[category][subcategory]
            result = self.classifier(
                text[:512],
                candidate_labels,
                multi_label=True
            )

            sorted_topics = sorted(
                zip(result["labels"], result["scores"]),
                key=lambda x: x[1],
                reverse=True
            )

            return [topic for topic, score in sorted_topics[:8] if score > topic_threshold]

        except Exception as e:
            print(f"Classification error: {str(e)}")
            return []