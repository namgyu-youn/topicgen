import os

import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .topic_list import TOPIC_LIST


class TopicAnalyzer:
    def __init__(self):
        # Initialize basic attributes
        self.device = "cpu"
        self.model_name = "roberta-base"
        self.onnx_path = "model.onnx"
        self.max_length = 1024

        # Set topic hierarchy before model initialization
        self.topic_hierarchy = TOPIC_LIST

        # Initialize tokenizer and ONNX session
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.session = self._load_onnx_model()

    def _get_num_labels(self) -> int:
        # Calculate number of topics for classification
        return len([topic for subcat in self.topic_hierarchy.values() for topic in subcat])

    def _convert_to_onnx(self) -> None:
        # Convert PyTorch model to ONNX format
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self._get_num_labels())
        model.eval()

        # Create sample input for ONNX export
        dummy_input = self.tokenizer("Sample text", return_tensors="pt")

        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            self.onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "output": {0: "batch_size"},
            },
            opset_version=14,
        )

    def _load_onnx_model(self) -> ort.InferenceSession:
        try:
            # Convert to ONNX if model doesn't exist
            if not os.path.exists(self.onnx_path):
                self._convert_to_onnx()

            # Initialize ONNX runtime session
            return ort.InferenceSession(self.onnx_path)
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise

    async def generate_topics(self, text: str, category: str, subcategory: str) -> list[dict[str, float]]:
        try:
            all_topics = [topic for subcat in self.topic_hierarchy[category].values() for topic in subcat]

            # Prepare input for inference
            inputs = self.tokenizer(text[: self.max_length], return_tensors="np", padding=True, truncation=True)

            # Run inference with ONNX
            onnx_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
            outputs = self.session.run(None, onnx_inputs)[0]

            # Process results
            probabilities = outputs[0]
            topics = [
                {"topic": topic, "score": float(score)} for topic, score in zip(all_topics, probabilities, strict=False) if score > 0.1
            ]

            return sorted(topics, key=lambda x: x["score"], reverse=True)[:10]

        except Exception as e:
            print(f"Error generating topics: {e!s}")
            return []
