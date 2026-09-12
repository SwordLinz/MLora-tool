"""
Standalone dataset toolset extracted from Kohya-MauveLinz.

Tabs:
- Dataset Tag Manager
- Batch Crop
- Video to Images
- RunningHub batch
"""
from __future__ import annotations

import argparse
import os

import gradio as gr

from app.class_gui_config import KohyaSSGUIConfig
from app.dataset_tag_manager_gui import gradio_dataset_tag_manager_tab
from app.batch_crop_gui import gradio_batch_crop_tab
from app.video_extract_gui import gradio_video_extract_tab
from app.runninghub_batch_gui import gradio_runninghub_batch_tab


def build_demo(headless: bool = False, config: KohyaSSGUIConfig | None = None) -> gr.Blocks:
    cfg = config or KohyaSSGUIConfig()
    with gr.Blocks() as demo:
        gr.Markdown("# Dataset Toolbox")
        with gr.Tab("Dataset Tag Manager"):
            gradio_dataset_tag_manager_tab(headless=headless, config=cfg)
        with gr.Tab("Batch Crop"):
            gradio_batch_crop_tab(headless=headless, config=cfg)
        with gr.Tab("Video to Images"):
            gradio_video_extract_tab(headless=headless, config=cfg)
        with gr.Tab("RunningHub batch"):
            gradio_runninghub_batch_tab(headless=headless, config=cfg)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset toolbox (tag manager / batch crop / video extract / RunningHub batch)")
    parser.add_argument("--headless", action="store_true", help="Hide browse/run buttons (for automated environments)")
    parser.add_argument("--config", default="./config.toml", help="Path to TOML config file")
    parser.add_argument("--listen", action="store_true", help="Listen on 0.0.0.0 instead of 127.0.0.1")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    args = parser.parse_args()

    # Config path is relative to the working directory, like the original GUI.
    config = KohyaSSGUIConfig(config_file_path=args.config)
    demo = build_demo(headless=args.headless, config=config)

    demo.launch(
        server_name="0.0.0.0" if args.listen else "127.0.0.1",
        server_port=args.port,
        inbrowser=not args.headless,
    )


if __name__ == "__main__":
    main()
