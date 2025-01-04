from transformers import pipeline
from typing import List, Dict
from topic_list import TOPIC_HIERARCHY

class TopicAnalyzer:
    def __init__(self):
        self.device = "cpu"
        self.set_classifier()
        self.max_length = 1024
        self.topic_hierarchy = TOPIC_HIERARCHY

    def set_device(self, device: str):
        if device != self.device:
            self.device = device
            self.set_classifier()

    def set_classifier(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="microsoft/deberta-v3-base",
            device=self.device,
            use_fast=False
        )

    async def generate_topics(self, text: str, category: str, subcategory: str) -> List[Dict]:
        try:
            all_topics = []
            for subcat in self.topic_hierarchy[category].values():
                all_topics.extend(subcat)

            result = self.classifier(
                text[:self.max_length],
                all_topics,
                multi_label=True
            )

            topics = [
                {"topic": topic, "score": score}
                for topic, score in zip(result["labels"], result["scores"])
                if score > 0.1
            ]

            return sorted(topics, key=lambda x: x["score"], reverse=True)[:10]

        except Exception as e:
            print(f"Error: {str(e)}")
            return []