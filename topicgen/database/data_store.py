import logging
from datetime import datetime
from typing import Any

from ..data_collector.models import RepositoryInfo
from .db_client import SQLiteClient

logger = logging.getLogger(__name__)

class DataStore:
    """Store for saving and retrieving repository and topic data."""

    def __init__(self, client=None):
        """
        Initialize the data store.

        Args:
            client: Optional SQLite client (creates a new one if not provided)
        """
        self.db = client or SQLiteClient().get_connection()

    async def save_repository(self, repo: RepositoryInfo) -> str:
        """
        Save repository information to the database.

        Args:
            repo: Repository information

        Returns:
            ID of the created or updated repository record
        """
        try:
            # Check if repository already exists by GitHub ID
            existing = await self.get_repository_by_github_id(repo.id)

            # Current timestamp for collection
            now = datetime.now().isoformat()

            repo_data = {
                "github_id": repo.id,
                "name": repo.name,
                "owner": repo.owner,
                "full_name": repo.full_name,
                "description": repo.description,
                "stars": repo.stars,
                "language": repo.language,
                "updated_at": now,
                "last_collected_at": now
            }

            if existing:
                # Update existing repository
                query = """
                    UPDATE repositories
                    SET name = :name,
                        owner = :owner,
                        full_name = :full_name,
                        description = :description,
                        stars = :stars,
                        language = :language,
                        updated_at = :updated_at,
                        last_collected_at = :last_collected_at
                    WHERE github_id = :github_id
                """
                self.db.execute(query, repo_data)
                self.db.commit()
                logger.debug(f"Updated repository: {repo.full_name}")
                return existing["id"]
            else:
                # Insert new repository
                repo_data["created_at"] = now
                query = """
                    INSERT INTO repositories (
                        github_id, name, owner, full_name,
                        description, stars, language, created_at, updated_at, last_collected_at
                    )
                    VALUES (
                        :github_id, :name, :owner, :full_name,
                        :description, :stars, :language, :created_at, :updated_at, :last_collected_at
                    )
                """
                cursor = self.db.execute(query, repo_data)
                self.db.commit()
                return cursor.lastrowid

        except Exception as e:
            logger.error(f"Error saving repository {repo.full_name}: {e!s}")
            raise

    async def save_topics(self, repository_id: str, topics: list[str]):
        """
        Save topics associated with a repository.

        Args:
            repository_id: ID of the repository in the database
            topics: List of topic names
        """
        try:
            if not topics:
                return

            # Prepare bulk insert data
            topic_data = [
                {
                    "repository_id": repository_id,
                    "name": topic.lower().strip(),
                    "created_at": datetime.now().isoformat()
                }
                for topic in topics
            ]

            # Use INSERT OR IGNORE for upsert behavior
            query = """
                INSERT OR IGNORE INTO topics (
                    repository_id, name, created_at
                )
                VALUES (
                    :repository_id, :name, :created_at
                )
            """

            # Execute in a transaction
            try:
                self.db.executemany(query, topic_data)
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

        except Exception as e:
            logger.error(f"Error saving topics for repository {repository_id}: {e!s}")
            raise

    async def save_repository_with_topics(self, repo: RepositoryInfo):
        """
        Save a repository and its topics in a single operation.

        Args:
            repo: Repository information

        Returns:
            UUID of the repository record
        """
        try:
            # Save repository
            repo_id = await self.save_repository(repo)

            # Save topics
            await self.save_topics(repo_id, repo.topics)

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
        query = "SELECT * FROM repositories WHERE github_id = ?"
        cursor = self.db.execute(query, (github_id,))
        result = cursor.fetchone()

        if result:
            return dict(result)
        return None

    async def get_repository_topics(self, repository_id: str) -> list[dict[str, Any]]:
        """
        Get topics for a specific repository.

        Args:
            repository_id: ID of the repository

        Returns:
            List of topic data
        """
        try:
            query = "SELECT * FROM topics WHERE repository_id = ?"
            cursor = self.db.execute(query, (repository_id,))
            results = cursor.fetchall()

            return [dict(row) for row in results]

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
            query = """
                SELECT name, COUNT(*) as count
                FROM topics
                GROUP BY name
                ORDER BY count DESC
                LIMIT ?
            """
            cursor = self.db.execute(query, (limit,))
            results = cursor.fetchall()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error getting popular topics: {e!s}")
            raise

    async def get_training_data(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get training data consisting of repository descriptions and their topics.

        Args:
            limit: Optional limit on the number of records to retrieve

        Returns:
            List of training data items with 'content' and 'labels' fields
        """
        try:
            query = """
                SELECT r.description AS content, GROUP_CONCAT(t.name, ', ') AS labels
                FROM repositories r
                JOIN topics t ON r.id = t.repository_id
                GROUP BY r.id
            """
            if limit:
                query += " LIMIT ?"
                cursor = self.db.execute(query, (limit,))
            else:
                cursor = self.db.execute(query)

            results = cursor.fetchall()
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error getting training data: {e!s}")
            raise

    async def get_last_collection_time(self) -> str | None:
        """
        Get the timestamp of the most recent data collection.

        Returns:
            ISO format timestamp of the last collection or None if no data exists
        """
        try:
            query = """
                SELECT MAX(last_collected_at) as last_collected
                FROM repositories
            """
            cursor = self.db.execute(query)
            result = cursor.fetchone()

            if result and result['last_collected']:
                return result['last_collected']
            return None

        except Exception as e:
            logger.error(f"Error getting last collection time: {e!s}")
            return None

    async def get_repositories_for_update(self, min_age_days: int = 7, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get repositories that need to be updated based on their last collection time.

        Args:
            min_age_days: Minimum age in days since last collection
            limit: Maximum number of repositories to return

        Returns:
            List of repository data that needs updating
        """
        try:
            # Calculate cutoff date
            from datetime import datetime, timedelta
            cutoff_date = (datetime.now() - timedelta(days=min_age_days.days if isinstance(min_age_days, timedelta) else min_age_days)).isoformat()

            query = """
                SELECT id, github_id, owner, name, full_name
                FROM repositories
                WHERE last_collected_at IS NULL OR last_collected_at < ?
                ORDER BY last_collected_at ASC
                LIMIT ?
            """

            cursor = self.db.execute(query, (cutoff_date, limit))
            results = cursor.fetchall()
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error getting repositories for update: {e!s}")
            raise

    async def mark_repositories_as_collected(self, repo_ids: list[int]):
        """
        Mark repositories as collected without updating other data.

        Args:
            repo_ids: List of repository IDs to mark as collected
        """
        try:
            if not repo_ids:
                return

            now = datetime.now().isoformat()

            # Update last_collected_at for all specified repositories
            query = """
                UPDATE repositories
                SET last_collected_at = ?
                WHERE id IN ({})
            """.format(','.join(['?'] * len(repo_ids)))

            params = [now] + repo_ids
            self.db.execute(query, params)
            self.db.commit()

            logger.debug(f"Marked {len(repo_ids)} repositories as collected")

        except Exception as e:
            logger.error(f"Error marking repositories as collected: {e!s}")
            raise
