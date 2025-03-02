import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to Python path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from topicgen.data_collector import TopicCollector
from topicgen.database import DataStore, SchemaManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def run_data_collection(
    min_stars: int = 1000,
    max_stars: int = 50000,
    language: str = "python",
    max_repos: int = 10000
):
    """
    Run the data collection pipeline to gather GitHub repository data.

    Args:
        min_stars: Minimum number of stars for repositories
        language: Programming language to filter repositories
        max_repos: Maximum number of repositories to collect
    """

    # Initialize components
    collector = TopicCollector()
    schema_manager = SchemaManager()
    data_store = DataStore()

    # Ensure database schema is set up
    logger.info("Initializing database schema")
    await schema_manager.initialize_schema()

    # Collect repositories and topics
    logger.info(f"Fetching up to {max_repos} repositories")
    repositories, topic_counts = await collector.collect_topics(
        min_stars=min_stars,
        languages=[language],  # Using just Python as requested
        max_repos=max_repos
    )

    # Get top topics
    top_topics = collector.get_top_topics(topic_counts, limit=50)
    logger.info(f"Top {len(top_topics)} topics: {', '.join([t[0] for t in top_topics[:10]])}")

    # Store repositories and topics in database
    logger.info(f"Storing {len(repositories)} repositories in database")
    for repo in repositories:
        await data_store.save_repository_with_topics(repo)

    # Analyze topic relationships
    logger.info("Analyzing topic relationships")
    topic_relations = await collector.analyze_topic_associations(repositories)
    logger.info(f"Analyzed relationships between {len(topic_relations)} topics")

    # Prepare training data (optional - can be done separately)
    logger.info("Pipeline execution completed successfully")
    return {
        "repositories_collected": len(repositories),
        "unique_topics": len(topic_counts),
        "top_topics": top_topics[:10]
    }

def main():
    """Command line entry point for data collection pipeline."""
    parser = argparse.ArgumentParser(description="GitHub Repository Data Collection Pipeline")
    parser.add_argument("--min-stars", type=int, default=1000, help="Minimum number of stars")
    parser.add_argument("--max-stars", type=int, default=50000, help="Maximum number of stars")
    parser.add_argument("--language", type=str, default="python", help="Programming language to filter by")
    parser.add_argument("--max-repos", type=int, default=1000, help="Maximum number of repositories to collect")

    args = parser.parse_args()

    # Run the pipeline
    try:
        result = asyncio.run(run_data_collection(
            min_stars=args.min_stars,
            language=args.language,
            max_repos=args.max_repos
        ))

        # Display results
        print("\n===== Data Collection Results =====")
        print(f"Repositories collected: {result['repositories_collected']}")
        print(f"Unique topics found: {result['unique_topics']}")
        print("Top 10 topics:")
        for topic, count in result['top_topics']:
            print(f"  - {topic}: {count} repositories")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e!s}")
        sys.exit(1)

if __name__ == "__main__":
    main()