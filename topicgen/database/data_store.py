import logging
from datetime import datetime
from typing import Any

from ..data_collector.repository_fetcher import RepositoryInfo
from .sqlite_client import SQLiteClient

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
                query = """
                    UPDATE repositories
                    SET name = :name,
                        owner = :owner,
                        full_name = :full_name,
                        url = :url,
                        description = :description,
                        stars = :stars,
                        language = :language,
                        updated_at = :updated_at
                    WHERE github_id = :github_id
                """
                self.db.execute(query, repo_data)
                self.db.commit()
                logger.debug(f"Updated repository: {repo.full_name}")
                return existing["id"]
            else:
                # Insert new repository
                repo_data["created_at"] = datetime.now().isoformat()
                query = """
                    INSERT INTO repositories (
                        github_id, name, owner, full_name, url,
                        description, stars, language, created_at, updated_at
                    )
                    VALUES (
                        :github_id, :name, :owner, :full_name, :url,
                        :description, :stars, :language, :created_at, :updated_at
                    )
                """
                cursor = self.db.execute(query, repo_data)
                self.db.commit()
                logger.debug(f"Inserted new repository: {repo.full_name}")
                return cursor.lastrowid

        except Exception as e:
            logger.error(f"Error saving repository {repo.full_name}: {e!s}")
            raise

    async def save_topics(self, repository_id: str, topics: list[str], source: str = "github"):
        """
        Save topics associated with a repository.

        Args:
            repository_id: ID of the repository in the database
            topics: List of topic names
            source: Source of the topics (default: 'github')
        """
        try:
            if not topics:
                return

            # Prepare bulk insert data
            topic_data = [
                {
                    "repository_id": repository_id,
                    "name": topic.lower().strip(),
                    "source": source,
                    "created_at": datetime.now().isoformat()
                }
                for topic in topics
            ]

            # Use INSERT OR IGNORE for upsert behavior
            query = """
                INSERT OR IGNORE INTO topics (
                    repository_id, name, source, created_at
                )
                VALUES (
                    :repository_id, :name, :source, :created_at
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
            query = "SELECT * FROM repositories WHERE github_id = ?"
            cursor = self.db.execute(query, (github_id,))
            result = cursor.fetchone()
            
            if result:
                return dict(result)
            return None

        except Exception as e:
            logger.error(f"Error getting repository by GitHub ID {github_id}: {e!s}")
            raise

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