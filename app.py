import logging
import traceback

import gradio as gr

from scripts.github_analyzer import GitHubAnalyzer
from scripts.topic_list import TOPIC_LIST

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    logger.info("Initializing GitHubAnalyzer...")
    analyzer = GitHubAnalyzer()
    logger.info("GitHubAnalyzer initialized successfully")
except Exception as e:
    logger.error(f"Error initializing GitHubAnalyzer: {e!s}")
    logger.error(traceback.format_exc())
    raise


async def process_url(url: str, main_cat: str, sub_cat: str, use_gpu: bool) -> tuple[str, str, str]:
    """Process GitHub URL and generate topics.

    Args:
        url: GitHub repository URL
        main_cat: Main category for classification
        sub_cat: Sub-category for classification
        use_gpu: Whether to use GPU for processing

    Returns:
        Tuple of (readme_topics, code_topics, dependencies)

    """
    try:
        logger.info(f"Processing URL: {url}")
        logger.info(f"Categories: main={main_cat}, sub={sub_cat}")
        logger.info(f"GPU enabled: {use_gpu}")

        if not all([url, main_cat, sub_cat]):
            logger.warning("Missing required inputs")
            return "Please select all categories", "", ""

        # Set device
        logger.info(f"Setting device to: {'cuda' if use_gpu else 'cpu'}")
        analyzer.set_device("cuda" if use_gpu else "cpu")

        # Analyze repository
        logger.info("Starting repository analysis...")
        response = await analyzer.analyze_repository(url, main_cat, sub_cat)

        if not response.success:
            error_msg = response.errors[0].message if response.errors else "Unknown error"
            logger.error(f"Analysis failed: {error_msg}")
            return error_msg, "", ""

        # Process results
        logger.info("Processing analysis results...")
        readme_topics = " ".join([f"#{topic['topic'].lower()}" for topic in response.data["readme_topics"]])
        code_topics = " ".join([f"#{topic['topic'].lower()}" for topic in response.data["code_topics"]])
        dependencies = " ".join([f"#{dep.lower()}" for dep in response.data["dependencies"]])

        logger.info("Analysis completed successfully")
        return readme_topics, code_topics, dependencies

    except Exception as e:
        logger.error(f"Error in process_url: {e!s}")
        logger.error(traceback.format_exc())
        return f"Error: {e!s}", "", ""


def create_interface():
    """Create and configure the Gradio interface."""
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
            logger.debug(f"Updating sub-categories for main category: {main_cat}")
            return gr.Dropdown(choices=list(TOPIC_LIST[main_cat].keys()) if main_cat else [])

        main_category.change(update_sub_category, inputs=main_category, outputs=sub_category)

        generate_btn.click(
            process_url,
            inputs=[url_input, main_category, sub_category, use_gpu],
            outputs=[readme_topics, code_topics, dependencies],
        )

    return demo


if __name__ == "__main__":
    try:
        logger.info("Starting Gradio interface...")
        demo = create_interface()
        demo.launch(share=True)
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e!s}")
        logger.error(traceback.format_exc())
        raise