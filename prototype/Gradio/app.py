import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * intensity

demo = gr.Interface(
    fn=greet,
    inputs=["text", gr.Slider(value=2, minimum=1, maximum=10, step=1)],
    outputs=[gr.Textbox(label="greeting", lines=3)],
    title="Demo title",
    description="Demo description.",
    examples=[
        ["Sulfyderz", 2],
    ],
    cache_examples=True,
    batch=True,
    max_batch_size=5,
)

demo.queue(
    default_concurrency_limit=5,
)

demo.launch()

