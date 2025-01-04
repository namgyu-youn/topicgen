from typing import Dict, List, Optional, Any
import aiohttp
from urllib.parse import urlparse
from .analyzer import TopicAnalyzer
from .error_handler import ErrorHandler

class GitHubAnalyzer:
    CORE_FILES = [
        'README.md',
        'requirements.txt',
        'pyproject.toml',
        'package.json',
        'main.py',
        'app.py',
        'train.py'
    ]

    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com"
        self.topic_analyzer = TopicAnalyzer()
        self.error_handler = ErrorHandler()

    def parse_github_url(self, url: str) -> tuple[str, str, str]:
        """Parse GitHub URL into components."""
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")

            if len(path_parts) < 2:
                return self.error_handler.handle_github_url_error(
                    url,
                    "URL must contain owner and repository"
                )

            owner = path_parts[0]
            repo = path_parts[1]
            branch = "main"  # default branch

            return owner, repo, branch
        except Exception as e:
            return self.error_handler.handle_github_url_error(url, str(e))

    async def _fetch_file(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a single file content."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                return self.error_handler.handle_file_fetch_error(
                    url,
                    f"HTTP {response.status}"
                )
        except Exception as e:
            return self.error_handler.handle_file_fetch_error(url, str(e))

    async def analyze_repository(
        self,
        repo_url: str,
        category: str,
        subcategory: str
    ) -> Dict[str, Any]:
        """Analyze repository and generate comprehensive topics."""
        try:
            files_content = await self._fetch_core_files(repo_url)
            if not files_content:
                return self.error_handler.handle_file_fetch_error(
                    repo_url,
                    "No core files found"
                )

            # Analyze README content
            readme_topics = []
            if 'README.md' in files_content:
                readme_topics = await self.topic_analyzer.generate_topics(
                    files_content['README.md'],
                    category,
                    subcategory
                )

            # Get dependencies
            dependencies = await self._analyze_dependencies(files_content)

            # Analyze Python files content
            code_content = ""
            for file in ['main.py', 'app.py', 'train.py']:
                if file in files_content:
                    code_content += files_content[file] + "\n"

            code_topics = []
            if code_content:
                code_topics = await self.topic_analyzer.generate_topics(
                    code_content,
                    category,
                    subcategory
                )

            return self.error_handler.success_response({
                "readme_topics": readme_topics,
                "code_topics": code_topics,
                "dependencies": dependencies
            })

        except Exception as e:
            return self.error_handler.handle_topic_analysis_error(
                str(e),
                {"repo_url": repo_url, "category": category, "subcategory": subcategory}
            )