"""GitHub Topic Generator - Automatic topic prediction for GitHub repositories."""

__version__ = "0.2.0"

from . import config, data_collector, database, models, pipelines

__all__ = ["config", "data_collector", "database", "models", "pipelines"]
