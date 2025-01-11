import functools
import time
import traceback
from collections.abc import Callable
from typing import Any, TypeVar

from .logger import get_logger

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def debug_trace(func: F) -> F:
    """Decorator to trace function calls with timing information.

    Args:
        func: Function to be decorated

    Returns:
        Decorated function with timing and logging
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        logger.debug(f"Entering {func_name}")
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"Exiting {func_name} (took {elapsed:.3f}s)")
            return result

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Exception in {func_name} (after {elapsed:.3f}s): {e!s}")
            logger.error(traceback.format_exc())
            raise

    return wrapper  # type: ignore


def debug_async_trace(func: F) -> F:
    """Decorator to trace async function calls with timing information.

    Args:
        func: Async function to be decorated

    Returns:
        Decorated async function with timing and logging
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        func_name = func.__name__
        logger.debug(f"Entering async {func_name}")
        start_time = time.perf_counter()

        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.debug(f"Exiting async {func_name} (took {elapsed:.3f}s)")
            return result

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"Exception in async {func_name} (after {elapsed:.3f}s): {e!s}")
            logger.error(traceback.format_exc())
            raise

    return wrapper  # type: ignore