import gradio as gr
from src.topic_generator.fetcher import GitHubFetcher
from src.topic_generator.analyzer import TopicAnalyzer

fetcher = GitHubFetcher()
analyzer = TopicAnalyzer()

async def process_url(url: str) -> str:
    try:
        readme_content = await fetcher.fetch_readme(url)
        topics = await analyzer.generate_topics(readme_content)
        return " ".join([f"#{topic}" for topic in topics])
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=process_url,
    inputs=gr.Textbox(
        label="GitHub URL",
        placeholder="Enter GitHub repository URL"
    ),
    outputs=gr.Textbox(label="Generated Topics"),
    title="GitHub Topic Generator",
    description="Generate topics from GitHub repository README files"
)

if __name__ == "__main__":
    demo.launch()
