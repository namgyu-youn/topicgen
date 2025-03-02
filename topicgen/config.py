import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_environment():
    """
    Load environment variables from .env file in project root.
    """
    # Find the project root directory (where .env should be located)
    root_dir = Path(__file__).parent.parent
    env_path = root_dir / '.env'

    # Load environment variables from .env file if it exists
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}")

# Call this at module import time
load_environment()

def get_env_var(var_name: str, default=None, required=False):
    """
    Get environment variable with validation.

    Args:
        var_name: Name of the environment variable
        default: Default value if not found
        required: Whether this variable is required

    Returns:
        Value of the environment variable or default

    Raises:
        ValueError: If variable is required but not found
    """
    value = os.environ.get(var_name, default)
    if required and value is None:
        logger.error(f"Required environment variable {var_name} not set")
        raise ValueError(f"Required environment variable {var_name} not set")
    return value

# Database configuration
DATABASE_PATH = get_env_var(
    "DATABASE_PATH",
    default=str(Path(__file__).parent.parent / "data" / "topicgen.db"),
    required=True
)