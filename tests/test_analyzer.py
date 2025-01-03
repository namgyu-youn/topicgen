import pytest
from scripts.analyzer import TopicAnalyzer

def test_topic_analyzer_initialization():
    analyzer = TopicAnalyzer()
    assert analyzer.device == "cpu"
    assert analyzer.max_length == 1024

@pytest.mark.asyncio
async def test_generate_topics():
    analyzer = TopicAnalyzer()
    topics = await analyzer.generate_topics(
        "Sample ML project with neural networks",
        "Data & AI",
        "Machine Learning"
    )
    assert isinstance(topics, list)
    assert all(isinstance(t, dict) for t in topics)