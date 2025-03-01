import logging
from datetime import datetime
from typing import Any

from ..data_collector.repository_fetcher import RepositoryInfo
from .supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

class DataStore:
    """Store for saving and retrieving repository and topic data."""

    def __init__(self, client=None):
        """
        Initialize the data store.

        Args:
            client: Optional Supabase client (creates a new one if not provided)
        """
        self.supabase = client or SupabaseClient().get_client()

    async def save_repository(self, repo: RepositoryInfo) -> str:
        """
        Save repository information to the database.

        Args:
            repo: Repository information

        Returns:
            UUID of the created or updated repository record
        """
        try:
            # Check if repository already exists by GitHub ID
            existing = await self.get_repository_by_github_id(repo.id)

            repo_data = {
                "github_id": repo.id,
                "name": repo.name,
                "owner": repo.owner,
                "full_name": repo.full_name,
                "url": repo.url,
                "description": repo.description,
                "stars": repo.stars,
                "language": repo.language,
                "updated_at": datetime.now().isoformat()
            }

            if existing:
                # Update existing repository
                result = await self.supabase.table("repositories") \
                    .update(repo_data) \
                    .eq("github_id", repo.id) \
                    .execute()

                logger.debug(f"Updated repository: {repo.full_name}")
                return existing["id"]
            else:
                # Insert new repository
                repo_data["created_at"] = datetime.now().isoformat()

                result = await self.supabase.table("repositories") \
                    .insert(repo_data) \
                    .execute()

                logger.debug(f"Inserted new repository: {repo.full_name}")
                return result.data[0]["id"]

        except Exception as e:
            logger.error(f"Error saving repository {repo.full_name}: {e!s}")
            raise

    async def save_topics(self, repository_id: str, topics: list[str], source: str = "github"):
        """
        Save topics associated with a repository.

        Args:
            repository_id: UUID of the repository in the database
            topics: List of topic names
            source: Source of the topics (default: 'github')
        """
        try:
            if not topics:
                # Early return if topics list is empty
                return

            # Prepare bulk insert data
            topic_data = [
                {
                    "repository_id": repository_id,
                    "name": topic.lower().strip(),  # Normalize topic names
                    "source": source,
                    "created_at": datetime.now().isoformat()
                }
                for topic in topics
            ]

            # Use upsert to handle conflicts
            await self.supabase.table("topics") \
                .upsert(topic_data, on_conflict=["repository_id", "name"]) \
                .execute()

        except Exception as e:
            logger.error(f"Error saving topics for repository {repository_id}: {e!s}")
            raise

    async def save_repository_with_topics(self, repo: RepositoryInfo, source: str = "github"):
        """
        Save a repository and its topics in a single operation.

        Args:
            repo: Repository information
            source: Source of the topics (default: 'github')

        Returns:
            UUID of the repository record
        """
        try:
            # Save repository
            repo_id = await self.save_repository(repo)

            # Save topics
            await self.save_topics(repo_id, repo.topics, source)

            return repo_id

        except Exception as e:
            logger.error(f"Error saving repository with topics {repo.full_name}: {e!s}")
            raise

    async def get_repository_by_github_id(self, github_id: int) -> dict[str, Any] | None:
        """
        Get repository by GitHub ID.

        Args:
            github_id: GitHub's repository ID

        Returns:
            Repository data or None if not found
        """
        try:
            result = await self.supabase.table("repositories") \
                .select("*") \
                .eq("github_id", github_id) \
                .execute()

            if result.data and len(result.data) > 0:
                return result.data[0]
            return None

        except Exception as e:
            logger.error(f"Error getting repository by GitHub ID {github_id}: {e!s}")
            raise

    async def get_repository_topics(self, repository_id: str) -> list[dict[str, Any]]:
        """
        Get topics for a specific repository.

        Args:
            repository_id: UUID of the repository

        Returns:
            List of topic data
        """
        try:
            result = await self.supabase.table("topics") \
                .select("*") \
                .eq("repository_id", repository_id) \
                .execute()

            return result.data

        except Exception as e:
            logger.error(f"Error getting topics for repository {repository_id}: {e!s}")
            raise

    async def get_most_popular_topics(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get the most popular topics based on usage count.

        Args:
            limit: Maximum number of topics to return

        Returns:
            List of topics with usage counts
        """
        try:
            # This would ideally be a SQL query that groups by topic name and counts occurrences
            # For demonstration, using a placeholder approach
            result = await self.supabase.rpc(
                'get_popular_topics',
                {'topic_limit': limit}
            ).execute()

            return result.data

        except Exception as e:
            logger.error(f"Error getting popular topics: {e!s}")
            raise

    async def get_training_data(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get training data consisting of repository descriptions and their topics.

        Args:
            limit: Optional limit on the number of records to retrieve

        Returns:
            List of training data items
        """
        try:
            # This would be a more complex query joining repositories and topics
            # Placeholder for demonstration
            result = await self.supabase.rpc(
                'get_training_data',
                {'data_limit': limit}
            ).execute()

            return result.data

        except Exception as e:
            logger.error(f"Error getting training data: {e!s}")
            raise