from fastapi import FastAPI, HTTPException
from topic_gen.fetcher import GitHubFetcher
from topic_gen.analyzer import TopicAnalyzer

app = FastAPI(title="GitHub Topic Generator")
fetcher = GitHubFetcher()
analyzer = TopicAnalyzer()

@app.post("/generate-topics")
async def generate_topics(url: str):
   try:
       readme_content = await fetcher.fetch_readme(url)
       topics = await analyzer.generate_topics(readme_content)
       return {"topics": [f"#{topic}" for topic in topics]}
   except Exception as e:
       raise HTTPException(status_code=400, detail=str(e))
