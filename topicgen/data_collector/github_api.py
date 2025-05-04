import base64
import logging
from typing import Any
from urllib.parse import urljoin

import aiohttp

from ..config import get_env_var

logger = logging.getLogger(__name__)

class GitHubAPIClient:
    """Client class for interacting with the GitHub API."""

    def __init__(self, token: str | None = None):
        """
        Initialize the GitHub API client.

        Args:
            token: GitHub API token (defaults to GITHUB_TOKEN env var)
        """
        # Get token from parameter or environment variable
        self.token = token or get_env_var("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }

        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
            logger.debug("GitHub API client initialized with token")
        else:
            logger.warning("GitHub API client initialized without token. Rate limits will be restricted.")

    async def _request(self, method: str, endpoint: str,
                      params: dict[str, Any] | None = None,
                      data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request to the GitHub API."""
        url = urljoin(self.base_url, endpoint)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(method, url, params=params,
                                         json=data, headers=self.headers) as response:
                    # Check rate limits
                    remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
                    if remaining < 10:
                        logger.warning(f"GitHub API rate limit almost reached. Remaining: {remaining}")

                    response_data = await response.json()

                    # Check for HTTP errors
                    if response.status >= 400:
                        error_msg = response_data.get("message", "Unknown error")
                        logger.error(f"GitHub API error ({response.status}): {error_msg}")
                        raise Exception(f"GitHub API error: {error_msg}")

                    return response_data

            except aiohttp.ClientError as e:
                logger.error(f"Request error: {e!s}")
                raise Exception(f"GitHub API communication failed: {e!s}") from e

    async def search_repositories(self, min_stars: int = 1000,
                                 max_stars: int = 50000,
                                 language: str | None = None,
                                 page: int = 1,
                                 per_page: int = 100) -> dict[str, Any]:
        """Search for repositories with a high number of stars."""
        endpoint = "/search/repositories"

        # Build query
        query = f"stars:{min_stars}..{max_stars}"
        if language:
            query += f" language:{language}"

        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "page": page,
            "per_page": per_page
        }

        return await self._request("GET", endpoint, params=params)

    async def get_repository_metadata(self, owner: str, repo: str) -> dict[str, Any]:
        """Get detailed information about a repository."""
        endpoint = f"/repos/{owner}/{repo}"
        return await self._request("GET", endpoint)

    async def get_repository_topics(self, owner: str, repo: str) -> list[str]:
        """Get topics for a repository."""
        endpoint = f"/repos/{owner}/{repo}/topics"
        response = await self._request("GET", endpoint)
        return response.get("names", [])

    async def get_repository_readme(self, owner: str, repo: str,
                                   branch: str = "main") -> str | None:
        """Get the README content of a repository."""
        try:
            endpoint = f"/repos/{owner}/{repo}/readme"
            response = await self._request("GET", endpoint)

            # Base64 decoding
            content = response.get("content", "")
            if content:
                content = base64.b64decode(content).decode("utf-8")
            return content
        except Exception as e:
            logger.warning(f"Failed to fetch README for {owner}/{repo}: {e!s}")
            return None
