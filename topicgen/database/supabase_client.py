import logging

from supabase import Client, create_client

from ..config import get_env_var

logger = logging.getLogger(__name__)

class SupabaseClient:
    """Singleton client for Supabase database interactions."""

    _instance = None

    def __new__(cls, url: str | None = None, key: str | None = None):
        """
        Create or return the singleton Supabase client instance.

        Args:
            url: Supabase project URL (defaults to SUPABASE_URL env var)
            key: Supabase API key (defaults to SUPABASE_KEY env var)

        Returns:
            SupabaseClient instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)


            # Get credentials from parameters or environment variables
            cls._instance.url = url or get_env_var("SUPABASE_URL", required=True)
            cls._instance.key = key or get_env_var("SUPABASE_KEY", required=True)

            # Debug logging (mask sensitive parts)
            masked_url = cls._instance.url[:10] + '***' if cls._instance.url else None
            masked_key = cls._instance.key[:5] + '***' + cls._instance.key[-5:] if cls._instance.key and len(cls._instance.key) > 10 else None
            logger.debug(f"Initializing Supabase client with URL: {masked_url} and key: {masked_key}")

            try:
                # Initialize Supabase client
                cls._instance.client = create_client(cls._instance.url, cls._instance.key)
                logger.info("Supabase client initialized successfully")

                # Quick validation test
                try:
                    # Attempt a simple query to validate connection
                    result = cls._instance.client.rpc('version').execute()
                    logger.info(f"Supabase connection validated: PostgreSQL {result.data}")
                except Exception as e:
                    logger.warning(f"Could not validate Supabase connection: {e!s}")

            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e!s}")
                raise

        return cls._instance

    def get_client(self) -> Client:
        """Get the Supabase client instance."""
        return self.client
