import logging
from collections import Counter
from typing import Any

from .github_api import GitHubAPIClient
from .repository_fetcher import RepositoryFetcher, RepositoryInfo

logger = logging.getLogger(__name__)

class TopicCollector:
    """Class for collecting topic information from GitHub repositories."""

    def __init__(self, api_client: GitHubAPIClient | None = None, token: str | None = None):
        """Initialize the topic collector."""
        self.api_client = api_client or GitHubAPIClient(token)
        self.repo_fetcher = RepositoryFetcher(api_client=self.api_client)

    async def collect_topics(self,
                            min_stars: int = 1000,
                            languages: list[str] | None = None,
                            max_repos: int = 500) -> tuple[list[RepositoryInfo], dict[str, int]]:
        """
        Collect topics from GitHub repositories.

        Returns:
            Tuple of (list of repository info, dictionary of topic frequencies)
        """
        # Fetch popular repositories
        repositories = await self.repo_fetcher.fetch_popular_repositories(
            min_stars=min_stars,
            languages=languages,
            max_repos=max_repos
        )

        # Collect all topics and their frequencies
        all_topics: Counter = Counter()
        for repo in repositories:
            for topic in repo.topics:
                all_topics[topic] += 1

        return repositories, dict(all_topics)

    def get_top_topics(self, topic_counts: dict[str, int], limit: int = 100) -> list[tuple[str, int]]:
        """Get the most frequently used topics."""
        return Counter(topic_counts).most_common(limit)

    async def analyze_topic_associations(self, repositories: list[RepositoryInfo]) -> dict[str, set[str]]:
        """
        Analyze how topics are related to each other.

        Returns:
            A mapping of topic names to sets of co-occurring topics
        """
        related_topics: dict[str, set[str]] = {}

        for repo in repositories:
            topics = repo.topics

            # For each topic in the current repository
            for topic in topics:
                if topic not in related_topics:
                    related_topics[topic] = set()

                # Add all other topics that co-occur with the current topic
                for related in topics:
                    if related != topic:
                        related_topics[topic].add(related)

        return related_topics

    async def prepare_training_data(self, repositories: list[RepositoryInfo]) -> list[dict[str, Any]]:
        """
        Prepare data for model training.

        Returns:
            List of training data (each item contains repository description and its topics)
        """
        training_data = []

        for repo in repositories:
            # Skip repositories without topics
            if not repo.topics:
                continue

            # Try to get README
            readme = await self.api_client.get_repository_readme(repo.owner, repo.name)

            # Prepare training data
            # Use both README content and description if available, otherwise just description
            content = ""
            if readme:
                content = f"{repo.description}\n\n{readme}"
            elif repo.description:
                content = repo.description
            else:
                # Skip if content is too minimal
                continue

            training_data.append({
                "repository_id": repo.id,
                "content": content,
                "topics": repo.topics
            })

        return training_data
