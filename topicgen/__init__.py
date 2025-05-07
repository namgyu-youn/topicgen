"""GitHub Topic Generator - Automatic topic prediction for GitHub repositories."""

__version__ = "0.2.0"

from . import data_collector, database, models, pipelines

__all__ = ["data_collector", "database", "models", "pipelines"]
