import pytest
from scripts.fetcher import GitHubFetcher

def test_parse_github_url():
   fetcher = GitHubFetcher()
   url = "https://github.com/Namgyu-Youn/repo"
   owner, repo, branch, file_path = fetcher.parse_github_url(url)
   assert owner == "Namgyu-Youn"
   assert repo == "github-topic-generator"
   assert branch == "main"
   assert file_path == "README.md"

@pytest.mark.asyncio
async def test_fetch_readme():
   fetcher = GitHubFetcher()
   with pytest.raises(Exception):
       await fetcher.fetch_readme("invalid_url")