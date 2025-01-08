import gradio as gr

from scripts.github_analyzer import GitHubAnalyzer
from scripts.topic_list import TOPIC_LIST

analyzer = GitHubAnalyzer()


async def process_url(url: str, main_cat: str, sub_cat: str, use_gpu: bool) -> tuple[str, str, str]:
    """
    Process GitHub URL and generate topics

    Args:
        url: GitHub repository URL
        main_cat: Main category for classification
        sub_cat: Sub-category for classification
        use_gpu: Whether to use GPU for processing

    Returns:
        Tuple of (readme_topics, code_topics, dependencies)

    """
    try:
        if not all([url, main_cat, sub_cat]):
            return "Please select all categories", "", ""

        analyzer.set_device("cuda" if use_gpu else "cpu")
        response = await analyzer.analyze_repository(url, main_cat, sub_cat)

        if not response.success:
            return response.errors[0].message, "", ""

        # Modified output format: only keywords without scores
        readme_topics = " ".join([f"#{topic['topic'].lower()}" for topic in response.data["readme_topics"]])

        code_topics = " ".join([f"#{topic['topic'].lower()}" for topic in response.data["code_topics"]])

        dependencies = " ".join([f"#{dep.lower()}" for dep in response.data["dependencies"]])

        return readme_topics, code_topics, dependencies

    except Exception as e:
        return f"Error: {e!s}", "", ""


def create_interface():
    """Create and configure the Gradio interface"""
    with gr.Blocks() as demo:
        gr.Markdown("# Enhanced GitHub Topic Generator")

        with gr.Row():
            url_input = gr.Textbox(label="GitHub URL", placeholder="Enter GitHub repository URL")

        with gr.Row():
            main_category = gr.Dropdown(choices=[None, *list(TOPIC_LIST.keys())], label="Main Category", value=None)
            sub_category = gr.Dropdown(choices=[], label="Sub Category")

        with gr.Row():
            use_gpu = gr.Checkbox(label="Use GPU (Check if you have CUDA-capable GPU)", value=False)

        with gr.Row():
            generate_btn = gr.Button("Generate Topics")

        with gr.Row():
            readme_topics = gr.Textbox(label="README Topics")
            code_topics = gr.Textbox(label="Code Analysis Topics")
            dependencies = gr.Textbox(label="Dependencies")

        def update_sub_category(main_cat):
            """Update sub-category choices based on main category selection"""
            return gr.Dropdown(choices=list(TOPIC_LIST[main_cat].keys()) if main_cat else [])

        main_category.change(update_sub_category, inputs=main_category, outputs=sub_category)

        generate_btn.click(
            process_url,
            inputs=[url_input, main_category, sub_category, use_gpu],
            outputs=[readme_topics, code_topics, dependencies],
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
