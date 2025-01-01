from typing import List, Dict

def format_topics(topics: List[str]) -> List[str]:
    return [f"#{topic.lower()}" for topic in topics]

def clean_text(text: str) -> str:
    return text.strip().lower()