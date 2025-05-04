"""Integration pipelines connecting the components of the GitHub Topic Generator."""

from .data_collection_pipeline import run_data_collection
from .model_training_pipeline import run_model_training

__all__ = ["run_data_collection", "run_model_training"]
