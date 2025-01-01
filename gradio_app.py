import gradio as gr
from topic_gen.fetcher import GitHubFetcher
from topic_gen.analyzer import TopicAnalyzer
from topic_gen.topic_hierarchy import TOPIC_HIERARCHY

fetcher = GitHubFetcher()
analyzer = TopicAnalyzer()

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
           specific_topics = gr.Dropdown(
               choices=[],
               label="Specific Topics",
               multiselect=True
           )

       generate_btn = gr.Button("Generate Topics")
       output = gr.Textbox(label="Generated Topics")

       def update_sub_category(main_cat):
           return gr.Dropdown(
               choices=list(TOPIC_HIERARCHY[main_cat].keys()) if main_cat else []
           )

       def update_specific_topics(main_cat, sub_cat):
           if main_cat and sub_cat:
               return gr.Dropdown(choices=TOPIC_HIERARCHY[main_cat][sub_cat])
           return gr.Dropdown(choices=[])

       async def generate_topics(url, main_cat, sub_cat, specific_topics):
           try:
               if not all([url, main_cat, sub_cat, specific_topics]):
                   return "Please select all categories"

               readme_content = await fetcher.fetch_readme(url)
               topics = await analyzer.generate_topics(
                   readme_content,
                   specific_topics
               )
               return " ".join([f"#{topic}" for topic in topics])
           except Exception as e:
               return f"Error: {str(e)}"

       main_category.change(
           update_sub_category,
           main_category,
           sub_category
       )
       sub_category.change(
           update_specific_topics,
           [main_category, sub_category],
           specific_topics
       )
       generate_btn.click(
           generate_topics,
           [url_input, main_category, sub_category, specific_topics],
           output
       )

   return demo

if __name__ == "__main__":
   demo = create_interface()
   demo.launch()