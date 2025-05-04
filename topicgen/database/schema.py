import logging
import sqlite3

from .db_client import SQLiteClient

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

            await self._migrate_existing_data()

            # Create tables
            self._create_repositories_table()
            self._create_topics_table()

            logger.info("Database schema initialization completed")
            return True
        except Exception as e:
            logger.error(f"Schema initialization error: {e!s}")
            raise

    async def _migrate_existing_data(self):
        """Migrate existing data to match the new schema."""
        try:
            try:
                # Check if repositories table exists
                cursor = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='repositories'")
                if cursor.fetchone():
                    # Table exists, check columns
                    cursor = self.db.execute("PRAGMA table_info(repositories)")
                    columns = [row['name'] for row in cursor.fetchall()]

                    # Add last_collected_at column if it doesn't exist
                    if 'last_collected_at' not in columns:
                        logger.info("Adding last_collected_at column to repositories table")
                        self.db.execute("ALTER TABLE repositories ADD COLUMN last_collected_at TEXT")

                    # Remove URL column if it exists (we now use owner/name instead)
                    if 'url' in columns:
                        logger.info("Migrating data from URL column to owner/name format")
                        # We can't drop columns in SQLite, so we'll handle this in the application layer

                # Check if topics table exists
                cursor = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topics'")
                if cursor.fetchone():
                    # Table exists, check columns
                    cursor = self.db.execute("PRAGMA table_info(topics)")
                    columns = [row['name'] for row in cursor.fetchall()]

                    # Remove source column if it exists
                    if 'source' in columns:
                        logger.info("Migrating topics table to remove source column")
                        # We can't drop columns in SQLite, so we'll handle this in the application layer

                # Commit changes
                self.db.commit()
                logger.info("Data migration completed successfully")

            except sqlite3.Error as e:
                logger.error(f"Error during data migration: {e}")
                raise

        except Exception as e:
            logger.error(f"Data migration error: {e!s}")
            raise

    def _create_repositories_table(self):
        """Create repositories table."""
        try:
            logger.info("Creating repositories table")

            # Create repositories table with updated schema
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    github_id INTEGER UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    description TEXT,
                    stars INTEGER NOT NULL,
                    forks INTEGER NOT NULL,
                    language TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    last_collected_at TEXT
                )
            """)

            # Create index on owner/name for faster lookups
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_repositories_owner_name
                ON repositories(owner, name)
            """)

            # Create index on last_collected_at for incremental collection
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_repositories_last_collected
                ON repositories(last_collected_at)
            """)

            self.db.commit()
            logger.info("Repositories table created or verified")

        except sqlite3.Error as e:
            logger.error(f"Error creating repositories table: {e}")
            raise

    def _create_topics_table(self):
        """Create topics table."""
        try:
            logger.info("Creating topics table")

            # Create topics table with updated schema
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    repository_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    FOREIGN KEY (repository_id) REFERENCES repositories(id) ON DELETE CASCADE,
                    UNIQUE(repository_id, name)
                )
            """)

            # Create index on topic name for faster lookups
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_topics_name
                ON topics(name)
            """)

            self.db.commit()
            logger.info("Topics table created or verified")

        except sqlite3.Error as e:
            logger.error(f"Error creating topics table: {e}")
            raise
