from dataclasses import dataclass

@dataclass
class RepositoryInfo:
    """Data class for storing repository information."""
    id: int
    name: str
    owner: str
    full_name: str
    description: str
    stars: int
    topics: list[str]
    created_at: str
    updated_at: str
    readme: str = ""
