"""
Standalone dataset toolset extracted from Kohya-MauveLinz.

Tabs:
- Dataset Tag Manager
- Crop (batch + single)
- Video to Images
- RunningHub batch
"""
from __future__ import annotations

import argparse
import os

import gradio as gr

from app.class_gui_config import KohyaSSGUIConfig
from app.dataset_tag_manager_gui import gradio_dataset_tag_manager_tab
from app.crop_gui import gradio_crop_tab
from app.video_extract_gui import gradio_video_extract_tab
from app.runninghub_batch_gui import gradio_runninghub_batch_tab

CUSTOM_CSS = """
/* Highlight the single-image save card so it stands out as the primary action. */
.sd-save-card {
    border: 1px solid var(--border-color-primary);
    border-left: 3px solid var(--color-accent, #ff7c00);
    border-radius: 10px;
    padding: 4px 12px 12px;
    background: var(--background-fill-secondary);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.18);
}
.sd-save-card .prose h4 {
    margin: 6px 0 2px;
}
.sd-save-status:empty {
    display: none;
}
/* Keep the crop preview from collapsing on narrow screens. */
.sd-save-card .file-preview {
    max-height: 80px;
}
"""


def build_demo(headless: bool = False, config: KohyaSSGUIConfig | None = None) -> gr.Blocks:
    cfg = config or KohyaSSGUIConfig()
    with gr.Blocks(css=CUSTOM_CSS) as demo:
        gr.Markdown("# 数据集工具箱")
        with gr.Tab("标签管理"):
            gradio_dataset_tag_manager_tab(headless=headless, config=cfg)
        with gr.Tab("裁剪"):
            gradio_crop_tab(headless=headless, config=cfg)
        with gr.Tab("视频抽帧"):
            gradio_video_extract_tab(headless=headless, config=cfg)
        with gr.Tab("RunningHub 批量"):
            gradio_runninghub_batch_tab(headless=headless, config=cfg)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset toolbox (tag manager / crop / video extract / RunningHub batch)")
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
