import gradio as gr
from src.rag import RAGPipeline

rag = RAGPipeline()

def answer_question(question: str, top_k: int, model: str):
    rag.ollama_model = model
    result = rag.query(question, k=top_k)

    sources_md = "\n\n".join([
        f"**{s['title']}** · score: `{s['score']}`\n\n> {s['snippet']}..."
        for s in result["sources"]
    ])
    return result["answer"], sources_md

with gr.Blocks(title="ragbench", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ragbench\nQ&A over 50,000 Wikipedia articles.")

    with gr.Row():
        with gr.Column(scale=3):
            question_box = gr.Textbox(
                label="Question",
                placeholder="e.g. What caused the French Revolution?",
                lines=2
            )
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top-K chunks")
            model_dropdown = gr.Dropdown(
                choices=["llama3.1:70b", "mistral"],
                value="llama3.1:70b",
                label="LLM"
            )

    submit_btn = gr.Button("Ask", variant="primary")
    answer_box = gr.Textbox(label="Answer", lines=5, interactive=False)
    sources_box = gr.Markdown(label="Retrieved Sources")

    submit_btn.click(
        fn=answer_question,
        inputs=[question_box, top_k_slider, model_dropdown],
        outputs=[answer_box, sources_box]
    )
    question_box.submit(
        fn=answer_question,
        inputs=[question_box, top_k_slider, model_dropdown],
        outputs=[answer_box, sources_box]
    )

if __name__ == "__main__":
    demo.launch(share=False)