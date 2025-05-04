import asyncio
import logging
from dataclasses import dataclass

from .github_api import GitHubAPIClient

logger = logging.getLogger(__name__)

@dataclass
class RepositoryInfo:
    """Data class for storing repository information."""
    id: int
    name: str
    owner: str
    full_name: str
    url: str
    description: str
    stars: int
    topics: list[str]
    created_at: str
    updated_at: str

class RepositoryFetcher:
    """Class for fetching popular repositories from GitHub."""

    def __init__(self, api_client: GitHubAPIClient | None = None, token: str | None = None):
        """
        Initialize the repository fetcher.
        """
        self.api_client = api_client or GitHubAPIClient(token)

    async def fetch_popular_repositories(self,
                                        min_stars: int = 1000,
                                        max_stars: int = 10000,
                                        max_repos: int = 500) -> list[RepositoryInfo]:
        """
        Fetch popular repositories from GitHub.
        """
        all_repos = []
        processed_repos: set[int] = set()  # Prevent duplicates

        page = 1
        while len(all_repos) < max_repos:
            try:
                search_result = await self.api_client.search_repositories(
                    min_stars=min_stars,
                    language="python",
                    page=page,
                    per_page=100  # GitHub API maximum
                )

                items = search_result.get("items", [])


                # Get additional info for each repository
                for item in items:
                    repo_id = item["id"]

                    # Skip already processed repositories
                    if repo_id in processed_repos:
                        continue

                    processed_repos.add(repo_id)

                    owner = item["owner"]["login"]
                    repo_name = item["name"]

                    # Get repository topics
                    topics = await self.api_client.get_repository_topics(owner, repo_name)

                    repo_info = RepositoryInfo(
                        id=repo_id,
                        name=repo_name,
                        owner=owner,
                        full_name=f"{owner}/{repo_name}",
                        url=item["html_url"],
                        description=item.get("description", "") or "",
                        stars=item["stargazers_count"],
                        topics=topics,
                        created_at=item["created_at"],
                        updated_at=item["updated_at"]
                    )

                    all_repos.append(repo_info)

                    # Stop if maximum number of repositories reached
                    if len(all_repos) >= max_repos:
                        break

                page += 1

                # Delay to respect GitHub API rate limits
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error fetching repositories: {e!s}")
                await asyncio.sleep(5)  # Wait before retrying after an error
                continue

        logger.info(f"Completed fetching {len(all_repos)} repositories")
        return all_repos
