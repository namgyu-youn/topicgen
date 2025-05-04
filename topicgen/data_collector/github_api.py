import aiohttp
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Optional
from urllib.parse import urljoin

from ..config import get_env_var

logger = logging.getLogger(__name__)

class GitHubAPICache:
    """Cache for GitHub API responses to reduce API calls."""

    def __init__(self, ttl: int = 3600):
        """
        Initialize the GitHub API cache.

        Args:
            ttl: Time-to-live in seconds for cache entries (default: 1 hour)
        """
        self.cache: dict[str, dict[str, Any]] = {}
        self.ttl = ttl
        logger.info(f"Initialized GitHub API cache with TTL of {ttl} seconds")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache if it exists and is not expired.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found or expired
        """
        if key in self.cache:
            item = self.cache[key]
            # Check if item is still valid
            if time.time() < item["expires"]:
                logger.debug(f"Cache hit for key: {key}")
                return item["data"]
            else:
                # Remove expired item
                logger.debug(f"Cache expired for key: {key}")
                del self.cache[key]
        return None

    async def set(self, key: str, data: Any, custom_ttl: Optional[int] = None):
        """
        Store item in cache with expiration.

        Args:
            key: Cache key
            data: Data to cache
            custom_ttl: Optional custom TTL for this specific item
        """
        ttl = custom_ttl if custom_ttl is not None else self.ttl
        self.cache[key] = {
            "data": data,
            "expires": time.time() + ttl
        }
        logger.debug(f"Cached data for key: {key} (expires in {ttl}s)")

    async def invalidate(self, key: str):
        """
        Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated cache for key: {key}")

    async def invalidate_pattern(self, pattern: str):
        """
        Invalidate all cache entries that match a pattern.

        Args:
            pattern: String pattern to match against cache keys
        """
        keys_to_remove = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.cache[key]
        logger.debug(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")

    async def clear(self):
        """Clear the entire cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        total_entries = len(self.cache)
        valid_entries = sum(1 for item in self.cache.values() if current_time < item["expires"])
        expired_entries = total_entries - valid_entries

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "memory_usage_bytes": self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> int:
        """
        Estimate memory usage of the cache in bytes.

        Returns:
            Estimated memory usage in bytes
        """
        import sys
        try:
            return sys.getsizeof(self.cache)
        except Exception:
            return 0


class SmartRateLimitHandler:
    """Handles GitHub API rate limits intelligently."""

    def __init__(self, min_buffer: int = 500):
        """
        Initialize the rate limit handler.

        Args:
            min_buffer: Minimum number of requests to keep in reserve
        """
        self.rate_limit_remaining = 5000  # Default GitHub API limit
        self.rate_limit_reset = 0
        self.min_rate_limit_buffer = min_buffer
        self.last_request_time = 0
        self.request_count = 0
        logger.info(f"Initialized rate limit handler with buffer of {min_buffer} requests")

    async def update_from_headers(self, headers: dict[str, str]):
        """
        Update rate limit information from response headers.

        Args:
            headers: Response headers from GitHub API
        """
        # Update rate limit information
        if 'X-RateLimit-Remaining' in headers:
            self.rate_limit_remaining = int(headers.get('X-RateLimit-Remaining', 0))

        if 'X-RateLimit-Reset' in headers:
            self.rate_limit_reset = int(headers.get('X-RateLimit-Reset', 0))

        # Log if rate limit is getting low
        if self.rate_limit_remaining < self.min_rate_limit_buffer:
            reset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.rate_limit_reset))
            logger.warning(
                f"Rate limit running low: {self.rate_limit_remaining} requests remaining. "
                f"Resets at {reset_time}"
            )

        # Update request tracking
        self.request_count += 1
        self.last_request_time = time.time()

    async def should_pause(self) -> bool:
        """
        Determine if requests should pause due to rate limit concerns.

        Returns:
            True if requests should pause, False otherwise
        """
        # If we're below the buffer threshold
        if self.rate_limit_remaining < self.min_rate_limit_buffer:
            current_time = time.time()

            # If reset time is in the future
            if current_time < self.rate_limit_reset:
                logger.warning(
                    f"Rate limit critical: {self.rate_limit_remaining} remaining. "
                    f"Pausing until reset."
                )
                return True

        return False

    async def wait_if_needed(self) -> bool:
        """
        Wait if rate limit requires it.

        Returns:
            True if waited, False otherwise
        """
        if await self.should_pause():
            wait_time = max(1, self.rate_limit_reset - time.time() + 5)  # Add 5s buffer
            logger.info(f"Rate limit approaching, pausing for {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
            return True
        return False

    async def adaptive_sleep(self, importance: str = "normal"):
        """
        Sleep for an adaptive duration based on rate limit and request importance.

        Args:
            importance: Importance of the request ("high", "normal", or "low")
        """
        # Calculate base delay based on remaining rate limit
        # As rate limit decreases, delay increases
        rate_limit_factor = max(0.1, 1 - (self.rate_limit_remaining / 5000))

        if importance == "high":
            # Minimal delay for high priority requests
            delay = 0.1 * rate_limit_factor
        elif importance == "normal":
            # Standard delay for normal requests
            delay = 0.5 * rate_limit_factor
        else:  # low importance
            # Longer delay for low priority requests
            delay = 2.0 * rate_limit_factor

        # Add small random jitter to prevent synchronized requests
        import random
        jitter = random.uniform(0, 0.2)
        final_delay = delay + jitter

        # Only log if delay is significant
        if final_delay > 0.2:
            logger.debug(f"Adaptive sleep: {final_delay:.2f}s for {importance} priority request")

        await asyncio.sleep(final_delay)

    async def get_status(self) -> dict[str, any]:
        """
        Get current rate limit status.

        Returns:
            Dictionary with rate limit status information
        """
        current_time = time.time()
        time_to_reset = max(0, self.rate_limit_reset - current_time)

        return {
            "remaining": self.rate_limit_remaining,
            "reset_timestamp": self.rate_limit_reset,
            "reset_in_seconds": time_to_reset,
            "request_count": self.request_count,
            "buffer_size": self.min_rate_limit_buffer,
            "status": "critical" if self.rate_limit_remaining < self.min_rate_limit_buffer else "normal"
        }


class GitHubAPIClient:
    """Client for interacting with the GitHub API."""

    def __init__(self, token: Optional[str] = None, use_cache: bool = True, cache_ttl: int = 3600):
        """
        Initialize the GitHub API client.

        Args:
            token: GitHub API token (optional)
            use_cache: Whether to use caching
            cache_ttl: Cache TTL in seconds
        """
        self.base_url = "https://api.github.com"
        self.token = token or get_env_var("GITHUB_TOKEN", "")
        self.use_cache = use_cache

        # Initialize cache and rate limiter
        self.cache = GitHubAPICache(ttl=cache_ttl)
        self.rate_limiter = SmartRateLimitHandler()

        logger.info(f"GitHub API client initialized with caching {'enabled' if use_cache else 'disabled'}")

    async def _generate_cache_key(self, method: str, endpoint: str,
                                 params: Optional[dict[str, Any]] = None,
                                 data: Optional[dict[str, Any]] = None) -> str:
        """
        Generate a unique cache key for a request.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body data

        Returns:
            Unique cache key
        """
        # Create a string representation of the request
        key_parts = [method.upper(), endpoint]

        if params:
            # Sort params to ensure consistent keys
            sorted_params = sorted(params.items())
            key_parts.append(json.dumps(sorted_params))

        if data:
            # Sort data to ensure consistent keys
            if isinstance(data, dict):
                sorted_data = json.dumps(data, sort_keys=True)
            else:
                sorted_data = str(data)
            key_parts.append(sorted_data)

        # Join parts and hash
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _request(self, method: str, endpoint: str,
                      params: Optional[dict[str, Any]] = None,
                      data: Optional[dict[str, Any]] = None,
                      cache_ttl: Optional[int] = None,
                      importance: str = "normal") -> dict[str, Any]:
        """
        Make a request to the GitHub API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            cache_ttl: Optional custom cache TTL
            importance: Request importance ("high", "normal", or "low")

        Returns:
            Response data
        """
        # Apply adaptive sleep based on importance
        await self.rate_limiter.adaptive_sleep(importance)

        # Check rate limits before making request
        await self.rate_limiter.wait_if_needed()

        # Check cache for GET requests
        if self.use_cache and method.upper() == "GET":
            cache_key = await self._generate_cache_key(method, endpoint, params, data)
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                return cached_data

        # Build URL
        url = urljoin(self.base_url, endpoint)

        # Set up headers
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }

        if self.token:
            headers["Authorization"] = f"token {self.token}"

        # Make request
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    headers=headers
                ) as response:
                    # Update rate limit info from headers
                    await self.rate_limiter.update_from_headers(response.headers)

                    # Parse response
                    response_data = await response.json()

                    # Check for HTTP errors
                    if response.status >= 400:
                        error_msg = response_data.get("message", "Unknown error")
                        logger.error(f"GitHub API error ({response.status}): {error_msg}")
                        raise Exception(f"GitHub API error: {error_msg}")

                    # Cache successful GET responses
                    if self.use_cache and method.upper() == "GET":
                        cache_key = await self._generate_cache_key(method, endpoint, params, data)
                        await self.cache.set(cache_key, response_data, custom_ttl=cache_ttl)

                    return response_data

            except aiohttp.ClientError as e:
                logger.error(f"Request error: {e!s}")
                raise Exception(f"GitHub API communication failed: {e!s}") from e

    async def search_repositories(self, min_stars: int = 1000,
                                 max_stars: int = 50000,
                                 language: str | None = None,
                                 page: int = 1,
                                 per_page: int = 100,
                                 pushed: str | None = None) -> dict[str, Any]:
        """
        Search for repositories with a high number of stars.

        Args:
            min_stars: Minimum number of stars
            max_stars: Maximum number of stars
            language: Programming language to filter by
            page: Page number for pagination
            per_page: Number of results per page
            pushed: Filter by last push date (e.g., ">=2020-01-01")
        """
        endpoint = "/search/repositories"

        # Build query
        query = f"stars:{min_stars}..{max_stars}"
        if language:
            query += f" language:{language}"
        if pushed:
            query += f" pushed:{pushed}"

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "page": page,
            "per_page": per_page
        }

        return await self._request("GET", endpoint, params=params)

    async def get_repository_metadata(self, owner: str, repo: str) -> dict[str, Any]:
        """Get metadata for a specific repository."""
        endpoint = f"/repos/{owner}/{repo}"
        return await self._request("GET", endpoint)

    async def get_repository_topics(self, owner: str, repo: str) -> list[str]:
        """Get topics for a specific repository."""
        endpoint = f"/repos/{owner}/{repo}/topics"
        response = await self._request("GET", endpoint)
        return response.get("names", [])

    async def get_repository_languages(self, owner: str, repo: str) -> dict[str, int]:
        """Get languages used in a repository."""
        endpoint = f"/repos/{owner}/{repo}/languages"
        return await self._request("GET", endpoint)

    async def get_rate_limit(self) -> dict[str, Any]:
        """Get current rate limit status."""
        endpoint = "/rate_limit"
        return await self._request("GET", endpoint, importance="high")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self.cache.get_stats()

    async def get_rate_limit_stats(self) -> dict[str, Any]:
        """Get rate limit statistics."""
        return await self.rate_limiter.get_status()
