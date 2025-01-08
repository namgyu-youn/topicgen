from urllib.parse import urlparse

import aiohttp


class GitHubFetcher:
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com"

    def parse_github_url(self, url: str) -> tuple[str, str, str, str]:
        """Parse GitHub URL into components: owner, repo, branch, file_path."""
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError("URL must include 'https://'")

        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL")

        owner = path_parts[0]
        repo = path_parts[1]
        branch = "main"
        file_path = "README.md"

        return owner, repo, branch, file_path

    async def fetch_readme(self, url: str) -> str:
        """
        Fetch README content from a GitHub repository.

        Args:
            url: GitHub repository URL

        Returns:
            The content of the README file

        Raises:
            Exception: If fetching fails for any reason

        """
        try:
            owner, repo, branch, file_path = self.parse_github_url(url)
            raw_url = f"{self.base_url}/{owner}/{repo}/{branch}/{file_path}"

            async with aiohttp.ClientSession() as session:
                async with session.get(raw_url) as response:
                    response.raise_for_status()
                    return await response.text()
        except Exception as e:
            raise Exception(f"Failed to fetch README: {e!s}") from e