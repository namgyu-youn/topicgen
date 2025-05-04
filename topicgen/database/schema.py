import logging

from .sqlite_client import SQLiteClient

logger = logging.getLogger(__name__)

class SchemaManager:
    """Manager for database schema creation and migrations."""

    def __init__(self, client=None):
        """
        Initialize the schema manager.

        Args:
            client: Optional SQLite client (creates a new one if not provided)
        """
        self.db = client or SQLiteClient().get_connection()

    async def initialize_schema(self):
        """Initialize the database schema with all required tables."""
        try:
            logger.info("Initializing database schema")

            # Create tables
            self._create_repositories_table()
            self._create_topics_table()

            logger.info("Database schema initialization completed")
            return True
        except Exception as e:
            logger.error(f"Schema initialization error: {e!s}")
            raise

    def _create_repositories_table(self):
        """Create the repositories table if it doesn't exist."""
        logger.info("Creating repositories table")

        # Define the SQL statements
        statements = [
            """
            CREATE TABLE IF NOT EXISTS repositories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                github_id INTEGER UNIQUE NOT NULL,
                name TEXT NOT NULL,
                owner TEXT NOT NULL,
                full_name TEXT NOT NULL,
                url TEXT NOT NULL,
                description TEXT,
                stars INTEGER,
                language TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_repositories_github_id ON repositories(github_id)",
            "CREATE INDEX IF NOT EXISTS idx_repositories_stars ON repositories(stars DESC)"
        ]

        try:
            for sql in statements:
                self.db.execute(sql)
                self.db.commit()
            logger.info("Repositories table created or verified")
        except Exception as e:
            logger.error(f"Error creating repositories table: {e!s}")
            raise

    def _create_topics_table(self):
        """Create the topics table if it doesn't exist."""
        logger.info("Creating topics table")

        # Define the SQL statements
        statements = [
            """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repository_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (repository_id) REFERENCES repositories(id)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_topics_name ON topics(name)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_topics_unique ON topics(repository_id, name)"
        ]

        try:
            for sql in statements:
                self.db.execute(sql)
                self.db.commit()
            logger.info("Topics table created or verified")
        except Exception as e:
            logger.error(f"Error creating topics table: {e!s}")
            raise
