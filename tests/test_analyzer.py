# tests/test_analyzer.py
import pytest
from topic_gen.analyzer import TopicAnalyzer

@pytest.mark.asyncio
async def test_generate_topics():
   analyzer = TopicAnalyzer()
   text = "This is a machine learning project using neural networks for image classification"
   topics = await analyzer.generate_topics(
       text=text,
       category="Data & AI",
       subcategory="Machine Learning",
       category_threshold=0.3,
       topic_threshold=0.5
   )
   assert len(topics) > 0
   assert all(isinstance(topic, str) for topic in topics)

# tests/test_fetcher.py
import pytest
from topic_gen.fetcher import GitHubFetcher

def test_parse_github_url():
   fetcher = GitHubFetcher()
   url = "https://github.com/owner/repo"
   owner, repo, branch, file_path = fetcher.parse_github_url(url)
   assert owner == "owner"
   assert repo == "repo"
   assert branch == "main"
   assert file_path == "README.md"

@pytest.mark.asyncio
async def test_fetch_readme():
   fetcher = GitHubFetcher()
   with pytest.raises(Exception):
       await fetcher.fetch_readme("invalid_url")