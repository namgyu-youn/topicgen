import asyncio
import logging
import sys
import time
from topicgen.data_collector import RepositoryCollector, TopicCollector
from topicgen.database import DataStore, SchemaManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# Define constants (hardcoded values)
MIN_STARS = MAX_REPOS = 1000
LANGUAGE = "python"

async def run_data_collection(language=LANGUAGE, min_stars=MIN_STARS, max_repos=MAX_REPOS):
    """
    Run the data collection pipeline to gather GitHub repository data.

    Args:
        language: Programming language to filter repositories
        min_stars: Minimum number of stars for repositories
        max_repos: Maximum number of repositories to collect

    Returns:
        Dictionary containing collection results summary
    """
    start_time = time.time()

    try:
        # Initialize components
        collector = RepositoryCollector()
        schema_manager = SchemaManager()
        data_store = DataStore()

        # Set up database schema
        logger.info("Initializing database schema")
        await schema_manager.initialize_schema()

        # Collect repositories and topics
        logger.info(f"Fetching up to {max_repos} repositories (language: {language}, min stars: {min_stars})")
        repositories, topic_counts = await collector.collect_topics(
            min_stars=min_stars,
            languages=[language] if language else None,
            max_repos=max_repos
        )

        # Get top topics
        top_topics = collector.get_top_topics(topic_counts, limit=50) if topic_counts else []
        if top_topics:
            logger.info(f"Top {len(top_topics)} topics: {', '.join([t[0] for t in top_topics[:10]])}")

        # Store repositories and topics in database
        if repositories:
            logger.info(f"Storing {len(repositories)} repositories in database")
            for repo in repositories:
                await data_store.save_repository_with_topics(repo)

        # Analyze topic relationships
        logger.info("Analyzing topic relationships")
        topic_relations = await collector.analyze_topic_associations(repositories)
        logger.info(f"Analyzed relationships between {len(topic_relations)} topics")

        # Calculate execution time
        execution_time = round(time.time() - start_time, 2)
        logger.info(f"Pipeline execution completed successfully ({execution_time} seconds)")

        # Collect API rate limit info (if available)
        rate_limit_info = getattr(collector.api_client, 'get_rate_limit_info', lambda: {})()

        return {
            "repositories_collected": len(repositories),
            "unique_topics": len(topic_counts),
            "top_topics": top_topics[:10],
            "execution_time": execution_time,
            "rate_limit": rate_limit_info.get('remaining', 'N/A')
        }

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return {
            "status": "failed",
            "error": str(e),
            "execution_time": round(time.time() - start_time, 2)
        }

def main():
    """Command line entry point for data collection pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="GitHub Repository Data Collection Pipeline")
    parser.add_argument("--language", type=str, default=LANGUAGE,
                      help=f"Programming language to filter by (default: {LANGUAGE})")
    parser.add_argument("--min-stars", type=int, default=MIN_STARS,
                      help=f"Minimum number of stars (default: {MIN_STARS})")
    parser.add_argument("--max-repos", type=int, default=MAX_REPOS,
                      help=f"Maximum number of repositories to collect (default: {MAX_REPOS})")

    args = parser.parse_args()

    # Run the pipeline
    try:
        result = asyncio.run(run_data_collection(
            language=args.language,
            min_stars=args.min_stars,
            max_repos=args.max_repos
        ))

        # Display results
        print("\n===== Data Collection Results =====")
        print(f"Repositories collected: {result['repositories_collected']}")
        print(f"Unique topics found: {result['unique_topics']}")
        print(f"Execution time: {result['execution_time']} seconds")
        print(f"GitHub API rate limit remaining: {result['rate_limit']}")

        if 'top_topics' in result and result['top_topics']:
            print("\nTop topics:")
            for topic, count in result['top_topics']:
                print(f"  - {topic}: {count} repositories")

        # Display cache statistics (if available)
        cache_stats = getattr(TopicCollector, 'get_cache_stats', lambda: None)()
        if cache_stats:
            print("\nCache Statistics:")
            print(f"  Valid entries: {cache_stats.get('valid_entries', 0)}")

        return 0
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
