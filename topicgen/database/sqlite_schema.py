import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SQLiteSchemaManager:
    """Manager for SQLite database schema initialization and migrations."""

    def __init__(self, db_path: str):
        """
        Initialize schema manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection = None

    async def initialize_schema(self):
        """Initialize the database schema."""
        try:
            # Create database directory if it doesn't exist
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row

            # Create tables
            self._create_repositories_table()
            self._create_topics_table()
            
            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}")
            raise

    def _create_repositories_table(self):
        """Create repositories table."""
        query = """
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
        """
        self.connection.execute(query)
        self.connection.commit()

    def _create_topics_table(self):
        """Create topics table."""
        query = """
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repository_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (repository_id) REFERENCES repositories(id)
            )
        """
        self.connection.execute(query)
        self.connection.commit()

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")