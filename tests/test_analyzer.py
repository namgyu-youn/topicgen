import pytest
from scripts.analyzer import TopicAnalyzer

@pytest.mark.asyncio
async def test_generate_topics():
   analyzer = TopicAnalyzer()
   text = "This is a machine learning project using neural networks for image classification"
   topics = await analyzer.generate_topics(
       text=text,
       category="Data & AI",
       subcategory="Machine Learning",
       category_threshold=0.1,
       topic_threshold=0.1
   )
   assert len(topics) > 0
   assert all(isinstance(topic, str) for topic in topics)