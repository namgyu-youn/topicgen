"""Integration pipelines connecting the components of the GitHub Topic Generator."""

from .data_collection_pipeline import run_data_collection

__all__ = ["run_data_collection"]
