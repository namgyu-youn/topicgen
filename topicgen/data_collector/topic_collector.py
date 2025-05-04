import asyncio
import logging
from collections import Counter
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from .github_api import GitHubAPIClient
from .models import RepositoryInfo

if TYPE_CHECKING:
    from ..database.data_store import DataStore

logger = logging.getLogger(__name__)

class RepositoryCollector:
    """Class for fetching and collecting repository and topic data from GitHub."""

    def __init__(self, api_client: GitHubAPIClient | None = None,
                 token: str | None = None,
                 data_store: "DataStore | None" = None):
        """
        Initialize the repository collector.

        Args:
            api_client: Optional GitHub API client
            token: Optional GitHub API token
            data_store: Optional data store for incremental collection
        """
        self.api_client = api_client or GitHubAPIClient(token)
        self.data_store = data_store

    async def fetch_popular_repositories(self,
                                        min_stars: int = 1000,
                                        max_stars: int = 10000,
                                        max_repos: int = 500,
                                        incremental: bool = False,
                                        update_days: int = 7) -> list[RepositoryInfo]:
        """
        Fetch popular repositories from GitHub.

        Args:
            min_stars: Minimum number of stars
            max_stars: Maximum number of stars
            max_repos: Maximum number of repositories to fetch
            incremental: Whether to use incremental collection
            update_days: Days since last update to consider for refresh

        Returns:
            List of repository information
        """
        all_repos = []
        processed_repos: set[int] = set()  # Prevent duplicates

        # If incremental mode and data store is available, get repositories to update
        if incremental and self.data_store:
            logger.info("Using incremental collection mode")

            # Get repositories that need updating
            repos_to_update = await self._get_repositories_for_update(update_days)

            if repos_to_update:
                logger.info(f"Found {len(repos_to_update)} repositories to update")

                # Process repositories in batches to avoid rate limiting
                batch_size = 10
                for i in range(0, len(repos_to_update), batch_size):
                    batch = repos_to_update[i:i+batch_size]

                    # Fetch repository data and topics in parallel
                    tasks = []
                    for repo in batch:
                        tasks.append(self._fetch_repository_data(repo['owner'], repo['name']))

                    # Wait for all tasks to complete
                    batch_results = await asyncio.gather(*tasks)

                    # Add valid results to the list
                    for repo in batch_results:
                        if repo and repo.id not in processed_repos:
                            all_repos.append(repo)
                            processed_repos.add(repo.id)

                    # Check if we've reached the maximum
                    if len(all_repos) >= max_repos:
                        logger.info(f"Reached maximum of {max_repos} repositories")
                        break

                    # Small delay to avoid rate limiting
                    await asyncio.sleep(1)

                logger.info(f"Completed incremental collection with {len(all_repos)} repositories")
                return all_repos
            else:
                logger.info("No repositories found for incremental update, falling back to regular collection")

        page = 1
        while len(all_repos) < max_repos:
            # Determine if we need to use pushed parameter for incremental collection
            pushed_param = None
            if incremental and self.data_store:
                last_collection_time = await self.data_store.get_last_collection_time()
                if last_collection_time:
                    if isinstance(last_collection_time, datetime):
                        date_str = last_collection_time.strftime('%Y-%m-%d')
                    else:
                        date_str = last_collection_time[:10]
                    pushed_param = f">={date_str}"
                    logger.info(f"Filtering by pushed date: {pushed_param}")

            try:
                search_results = await self.api_client.search_repositories(
                    min_stars=min_stars,
                    max_stars=max_stars,
                    language="python",
                    page=page,
                    pushed=pushed_param
                )

                for item in search_results.get("items", []):
                    if item["id"] in processed_repos:
                        continue

                    repo_info = await self._process_repository(item)
                    if repo_info:
                        all_repos.append(repo_info)
                        processed_repos.add(item["id"])

                        if len(all_repos) % 10 == 0:
                            logger.info(f"Collected {len(all_repos)} repositories so far")

                    if len(all_repos) >= max_repos:
                        logger.info(f"Reached maximum of {max_repos} repositories")
                        break

                page += 1
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error fetching repositories: {e!s}")
                break

        logger.info(f"Collected {len(all_repos)} repositories in total")
        return all_repos

    async def _get_repositories_for_update(self, update_days: int) -> list[dict]:
        """
        Get repositories that need updating.

        Args:
            update_days: Days since last update to consider for refresh

        Returns:
            List of repositories to update
        """
        if not self.data_store:
            return []

        # Get repositories to update
        return await self.data_store.get_repositories_for_update(update_days)

    async def _fetch_repository_data(self, owner: str, name: str) -> Optional[RepositoryInfo]:
        """
        Fetch repository data and topics.

        Args:
            owner: Repository owner
            name: Repository name

        Returns:
            Repository information or None if error
        """
        try:
            # Fetch repository metadata
            repo_data = await self.api_client.get_repository_metadata(owner, name)

            # Fetch repository topics
            topics = await self.api_client.get_repository_topics(owner, name)

            # Create repository info object
            return RepositoryInfo(
                id=repo_data["id"],
                name=repo_data["name"],
                owner=repo_data["owner"]["login"],
                full_name=repo_data["full_name"],
                description=repo_data.get("description", ""),
                stars=repo_data["stargazers_count"],
                topics=topics,
                created_at=repo_data["created_at"],
                updated_at=repo_data["updated_at"]
            )
        except Exception as e:
            logger.error(f"Error fetching repository data for {owner}/{name}: {e!s}")
            return None

    async def _process_repository(self, repo_data: dict) -> Optional[RepositoryInfo]:
        """
        Process repository data from search results.

        Args:
            repo_data: Repository data from search results

        Returns:
            Repository information or None if error
        """
        try:
            owner = repo_data["owner"]["login"]
            name = repo_data["name"]

            # Fetch repository topics
            topics = await self.api_client.get_repository_topics(owner, name)

            # Create repository info object
            return RepositoryInfo(
                id=repo_data["id"],
                name=name,
                owner=owner,
                full_name=repo_data["full_name"],
                description=repo_data.get("description", ""),
                stars=repo_data["stargazers_count"],
                topics=topics,
                created_at=repo_data["created_at"],
                updated_at=repo_data["updated_at"]
            )
        except Exception as e:
            logger.error(f"Error processing repository: {e!s}")
            return None

    async def collect_topics(self,
                            min_stars: int = 1000,
                            languages: str = "python",
                            max_repos: int = 500,
                            incremental: bool = False,
                            update_days: int = 7) -> tuple[list[RepositoryInfo], dict[str, int]]:
        """
        Collect topics from GitHub repositories.

        Args:
            min_stars: Minimum number of stars
            languages: List of languages to filter by
            max_repos: Maximum number of repositories to fetch
            incremental: Whether to use incremental collection
            update_days: Days since last update to consider for refresh

        Returns:
            Tuple of (list of repository info, dictionary of topic frequencies)
        """
        # Fetch popular repositories
        repositories = await self.fetch_popular_repositories(
            min_stars=min_stars,
            max_repos=max_repos,
            incremental=incremental,
            update_days=update_days
        )

        # Collect all topics and their frequencies
        all_topics: Counter = Counter()
        for repo in repositories:
            for topic in repo.topics:
                all_topics[topic] += 1

        return repositories, dict(all_topics)

    def get_top_topics(self, topic_counts: dict[str, int], limit: int = 100) -> list[tuple[str, int]]:
        """
        Get the most frequently used topics.

        Args:
            topic_counts: Dictionary of topic counts
            limit: Maximum number of topics to return

        Returns:
            List of (topic, count) tuples
        """
        return Counter(topic_counts).most_common(limit)

    async def analyze_topic_associations(self, repositories: list[RepositoryInfo]) -> dict[str, set[str]]:
        """
        Analyze how topics are related to each other.

        Args:
            repositories: List of repository information

        Returns:
            Dictionary mapping topics to sets of related topics
        """
        # Map each topic to a set of related topics
        topic_associations: dict[str, set[str]] = {}

        # Process each repository
        for repo in repositories:
            # Skip repositories with fewer than 2 topics
            if len(repo.topics) < 2:
                continue

            # For each topic, add all other topics as related
            for topic in repo.topics:
                if topic not in topic_associations:
                    topic_associations[topic] = set()

                # Add all other topics as related
                for related_topic in repo.topics:
                    if related_topic != topic:
                        topic_associations[topic].add(related_topic)

        logger.info(f"Analyzed relationships between {len(topic_associations)} topics")
        return topic_associations
