from typing import Dict, List, Optional
import aiohttp
from urllib.parse import urlparse
from .analyzer import TopicAnalyzer

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

    def parse_github_url(self, url: str) -> tuple[str, str, str]:
        """Parse GitHub URL into components."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 2:
            raise ValueError("Invalid GitHub URL")

        owner = path_parts[0]
        repo = path_parts[1]
        branch = "main"  # default branch

        return owner, repo, branch

    async def _fetch_file(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch a single file content."""
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                return None
        except Exception:
            return None

    async def _fetch_core_files(self, repo_url: str) -> Dict[str, str]:
        """Fetch content of core files from repository."""
        owner, repo, branch = self.parse_github_url(repo_url)
        files_content = {}

        async with aiohttp.ClientSession() as session:
            for file in self.CORE_FILES:
                url = f"{self.base_url}/{owner}/{repo}/{branch}/{file}"
                content = await self._fetch_file(session, url)
                if content:
                    files_content[file] = content

        return files_content

    async def _analyze_dependencies(self, files_content: Dict[str, str]) -> List[str]:
        """Extract dependencies from requirement files."""
        deps = set()

        if 'requirements.txt' in files_content:
            for line in files_content['requirements.txt'].split('\n'):
                if line and not line.startswith('#'):
                    package = line.split('==')[0].split('>=')[0].strip()
                    deps.add(package)

        if 'pyproject.toml' in files_content:
            # Basic TOML parsing for dependencies
            content = files_content['pyproject.toml']
            if '[tool.poetry.dependencies]' in content:
                deps.update(self._parse_poetry_deps(content))

        if 'package.json' in files_content:
            try:
                import json
                pkg_json = json.loads(files_content['package.json'])
                deps.update(pkg_json.get('dependencies', {}).keys())
                deps.update(pkg_json.get('devDependencies', {}).keys())
            except json.JSONDecodeError:
                pass

        return list(deps)

    def _parse_poetry_deps(self, content: str) -> List[str]:
        """Parse dependencies from pyproject.toml."""
        deps = set()
        in_deps = False

        for line in content.split('\n'):
            if '[tool.poetry.dependencies]' in line:
                in_deps = True
            elif in_deps and line.startswith('['):
                in_deps = False
            elif in_deps and '=' in line:
                package = line.split('=')[0].strip()
                deps.add(package)

        return list(deps)

    async def analyze_repository(
        self,
        repo_url: str,
        category: str,
        subcategory: str
    ) -> Dict[str, List]:
        """Analyze repository and generate comprehensive topics."""
        files_content = await self._fetch_core_files(repo_url)

        if not files_content:
            return {"error": "No core files found"}

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

        return {
            "readme_topics": readme_topics,
            "code_topics": code_topics,
            "dependencies": dependencies
        }