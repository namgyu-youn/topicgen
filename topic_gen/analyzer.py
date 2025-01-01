from transformers import pipeline
from typing import List

class TopicAnalyzer:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification")

    async def generate_topics(self, text: str, selected_topics: List[str]) -> List[str]:
        if not selected_topics:
            return []

        result = self.classifier(
            text[:512],
            selected_topics,
            multi_label=True
        )

        sorted_topics = sorted(
            zip(result["labels"], result["scores"]),
            key=lambda x: x[1],
            reverse=True
        )

        return [topic for topic, score in sorted_topics[:10] if score > 0.6]