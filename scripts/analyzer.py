from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
from .topic_list import TOPIC_LIST

class TopicAnalyzer:
    def __init__(self):
        self.device = "cpu"
        self.model_name = "microsoft/deberta-v3-base"
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.max_length = 1024
        self.topic_hierarchy = TOPIC_LIST
        self.set_classifier()

    def set_device(self, device: str):
        if device != self.device:
            self.device = device
            self.set_classifier()

    def set_classifier(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            ).to(self.device)

            # Set zero-shot pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
        except Exception as e:
            print(f"Error initializing classifier: {str(e)}")
            raise

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
            print(f"Error generating topics: {str(e)}")
            return []