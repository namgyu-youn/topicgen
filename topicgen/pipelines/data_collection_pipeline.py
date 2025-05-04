import asyncio
import sys
import time
from pathlib import Path

# Add project root to Python path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from topicgen.data_collector.topic_collector import RepositoryCollector
from topicgen.data_collector.github_api import GitHubAPIClient
from topicgen.database.schema import SchemaManager
from topicgen.database.data_store import DataStore
from topicgen.utils.cli import setup_logging, get_data_collection_parser

logger = setup_logging()

async def run_data_collection(
    min_stars: int = 1000,
    max_stars: int = 50000,
    language: str = "python",
    max_repos: int = 10000,
    incremental: bool = True,
    update_days: int = 7,
    cache_ttl: int = 3600,
):
    """
    Run the data collection pipeline to gather GitHub repository data.

    Args:
        min_stars: Minimum number of stars for repositories
        max_stars: Maximum number of stars for repositories
        language: Programming language to filter repositories
        max_repos: Maximum number of repositories to collect
        incremental: Whether to use incremental collection
        update_days: Days since last update to consider for refresh
        cache_ttl: Cache TTL in seconds
    """
    start_time = time.time()

    # Initialize database components
    schema_manager = SchemaManager()
    data_store = DataStore()

    # Ensure database schema is set up
    await schema_manager.initialize_schema()

    # Initialize API client (mock or real)
    api_client = GitHubAPIClient(use_cache=True, cache_ttl=cache_ttl)
    logger.info("Using real GitHub API client")

    # Initialize repository collector
    repo_collector = RepositoryCollector(
        api_client=api_client,
        data_store=data_store
    )

    # Collect repositories and topics
    logger.info(f"Starting data collection with {'incremental' if incremental else 'full'} mode")
    repositories, topic_counts = await repo_collector.collect_topics(
        min_stars=min_stars,
        languages=[language],
        max_repos=max_repos,
        incremental=incremental,
        update_days=update_days
    )

    # Get top topics
    #top_topics = collector.get_top_topics(topic_counts, limit=50)
    #logger.info(f"Top {len(top_topics)} topics: {', '.join([t[0] for t in top_topics[:10]])}")

    # Store repositories in database
    logger.info(f"Storing {len(repositories)} repositories in database")
    for repo in repositories:
        await data_store.save_repository(repo)

    # Analyze topic relationships
    logger.info("Analyzing topic relationships")
    topic_relationships = await repo_collector.analyze_topic_associations(repositories)
    logger.info(f"Analyzed relationships between {len(topic_relationships)} topics")

    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(f"Pipeline execution completed successfully in {execution_time:.2f} seconds")

    # Get cache and rate limit statistics
    cache_stats = await api_client.get_cache_stats()
    rate_limit_stats = await api_client.get_rate_limit_stats()

    # Print summary
    print("\n===== Data Collection Results =====")
    print(f"Repositories collected: {len(repositories)}")
    print(f"Unique topics found: {len(topic_counts)}")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"GitHub API rate limit remaining: {rate_limit_stats['remaining']}")
    print()
    print("Cache Statistics:")
    print(f"  Valid entries: {cache_stats['valid_entries']}")
    print()

def main():
    """Main entry point for the data collection pipeline."""
    parser = get_data_collection_parser()
    args = parser.parse_args()

    # Run the pipeline
    asyncio.run(run_data_collection(
        min_stars=args.min_stars,
        max_stars=args.max_stars,
        language=args.language,
        max_repos=args.max_repos,
        incremental=args.incremental,
        update_days=args.update_days,
        cache_ttl=args.cache_ttl
    ))

if __name__ == "__main__":
    main()
