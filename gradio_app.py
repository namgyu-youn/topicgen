import gradio as gr
from topic_gen.fetcher import GitHubFetcher
from topic_gen.analyzer import TopicAnalyzer
from topic_gen.topic_hierarchy import TOPIC_HIERARCHY
from topic_gen.utils import format_topics

fetcher = GitHubFetcher()

async def process_url(
    url: str,
    main_cat: str,
    sub_cat: str,
    category_threshold: float,
    topic_threshold: float
) -> tuple[str, str]:
    """Process GitHub URL and generate topics."""
    try:
        if not all([url, main_cat, sub_cat]):
            return "Please select all categories", ""

        readme_content = await fetcher.fetch_readme(url)
        analyzer = TopicAnalyzer()
        topics = await analyzer.generate_topics(
            text=readme_content,
            category=main_cat,
            subcategory=sub_cat,
            category_threshold=category_threshold,
            topic_threshold=topic_threshold
        )

        if not topics:
            return "No relevant topics found for the selected categories", ""

        generated_topics = " ".join(format_topics(topics))
        recommended_topics = " ".join([
            f"#{topic.replace(' ', '-').lower()}"
            for topic in TOPIC_HIERARCHY[main_cat][sub_cat]
        ])

        return generated_topics, recommended_topics

    except Exception as e:
        return f"Error: {str(e)}", ""

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# GitHub Topic Generator")

        with gr.Row():
            url_input = gr.Textbox(
                label="GitHub URL",
                placeholder="Enter GitHub repository URL"
            )

        with gr.Row():
            main_category = gr.Dropdown(
                choices=list(TOPIC_HIERARCHY.keys()),
                label="Main Category"
            )
            sub_category = gr.Dropdown(
                choices=[],
                label="Sub Category"
            )

        with gr.Row():
            category_threshold = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.4,
                step=0.1,
                label="Category Relevance Threshold",
                info="Minimum confidence score for main category (default: 0.4)"
            )
            topic_threshold = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.6,
                step=0.1,
                label="Topic Confidence Threshold",
                info="Minimum confidence score for generated topics (default: 0.6)"
            )

        with gr.Row():
            generate_btn = gr.Button("Generate Topics")

        with gr.Row():
            output = gr.Textbox(label="Generated Topics")
            recommended = gr.Textbox(label="Recommended Topics")

        def update_sub_category(main_cat):
            return gr.Dropdown(
                choices=list(TOPIC_HIERARCHY[main_cat].keys()) if main_cat else []
            )

        main_category.change(
            update_sub_category,
            inputs=main_category,
            outputs=sub_category
        )

        generate_btn.click(
            process_url,
            inputs=[
                url_input,
                main_category,
                sub_category,
                category_threshold,
                topic_threshold
            ],
            outputs=[output, recommended]
        )

    return demo

if __name__ == "__main__":
   demo = create_interface()
   demo.launch()