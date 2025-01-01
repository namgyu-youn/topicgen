import requests
from urllib.parse import urlparse

class GitHubFetcher:
    def __init__(self):
        self.base_url = "https://github.com/Namgyu-Youn/tag-generator.git"

    def parse_github_url(self, url: str) -> tuple[str, str, str, str]:
        """Parse GitHub URL into components."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL")

        owner = path_parts[0]
        repo = path_parts[1]
        branch = "main"
        file_path = "README.md"

        if len(path_parts) > 4 and path_parts[2] == "blob":
            branch = path_parts[3]
            file_path = "/".join(path_parts[4:])

        return owner, repo, branch, file_path

    async def fetch_readme(self, url: str) -> str:
        """Fetch README content from GitHub URL."""
        try:
            owner, repo, branch, file_path = self.parse_github_url(url)
            raw_url = f"{self.base_url}/{owner}/{repo}/{branch}/{file_path}"

            response = requests.get(raw_url)
            response.raise_for_status()
            return response.text

        except requests.RequestException as e:
            raise Exception(f"Failed to fetch README: {str(e)}")