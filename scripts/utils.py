from typing import list


def format_topics(topics: list[str]) -> list[str]:
    return [f"#{topic.lower()}" for topic in topics]


def clean_text(text: str) -> str:
    return text.strip().lower()
