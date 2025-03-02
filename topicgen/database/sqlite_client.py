import logging
import sqlite3
from pathlib import Path

from ..config import get_env_var

logger = logging.getLogger(__name__)

class SQLiteClient:
    """Client for SQLite database interactions."""

    def __init__(self, db_path: str | None = None):
        """
        Initialize SQLite client.
        
        Args:
            db_path: Path to SQLite database file (defaults to DATABASE_PATH env var)
        """
        self.db_path = db_path or get_env_var("DATABASE_PATH", required=True)
        
        # Create database directory if it doesn't exist
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        
        logger.info(f"SQLite client initialized with database at {self.db_path}")

    def get_connection(self) -> sqlite3.Connection:
        """Get the SQLite connection."""
        return self.connection

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            SQLite cursor
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            return cursor
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            raise

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("SQLite connection closed")