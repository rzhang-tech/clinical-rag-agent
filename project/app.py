import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import gradio as gr
import uvicorn

from api.main import create_app
from ui.gradio_app import create_gradio_ui
from ui.css import custom_css

# Build FastAPI app
app = create_app()

# Mount Gradio UI at /ui — shares the same process as the API
demo = create_gradio_ui()
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
