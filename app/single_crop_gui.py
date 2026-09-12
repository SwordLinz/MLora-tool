"""
Single-image crop tab — interactive crop with the Gradio ImageEditor
(crop / resize transforms), optional aspect-ratio lock, optional target
resize after cropping, and transparent-background flattening on save.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from .class_gui_config import KohyaSSGUIConfig
from .common_gui import get_folder_path, scriptdir
from .custom_logging import setup_logging

log = setup_logging()

RATIO_CHOICES = ["自由", "1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "4:5", "5:4"]
RATIO_FREE = "自由"
FIT_STRETCH = "拉伸"
FIT_COVER = "裁剪补齐（溢出部分裁掉）"
FIT_CONTAIN = "留白填充（白边）"


def _parse_ratio(choice: str) -> Optional[float]:
    """Return width/height as a float, or None for '自由'."""
    if not choice or choice == RATIO_FREE:
        return None
    try:
        w_s, h_s = choice.split(":", 1)
        w, h = float(w_s), float(h_s)
        if w <= 0 or h <= 0:
            return None
        return w / h
    except ValueError:
        return None


def _flatten_transparency_to_white(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA"):
        if img.mode == "LA":
            img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _crop_from_editor_value(editor_value: dict) -> Optional[Image.Image]:
    """Extract the crop result from an ImageEditor value dict (background + composite)."""
    if not isinstance(editor_value, dict):
        return None
    bg = editor_value.get("background")
    composite = editor_value.get("composite")

    # Prefer the composite: it reflects the applied crop transform.
    base = composite if composite is not None else bg
    if base is None:
        return None
    if isinstance(base, np.ndarray):
        base = Image.fromarray(base)

    # When a crop rect is applied the composite equals the cropped region,
    # but it is placed on a canvas-sized transparent image in some versions.
    # Trim fully-transparent borders to recover the actual pixels.
    if base.mode in ("RGBA", "LA"):
        alpha = base.split()[3]
        bbox = alpha.getbbox()
        if bbox:
            base = base.crop(bbox)

    if bg is not None and composite is not None:
        # If nothing was cropped, composite still matches the background:
        # prefer the trimmed result either way, it is identical.
        pass
    return base


def _crop_by_ratio(img: Image.Image, ratio: Optional[float]) -> Image.Image:
    """Center-crop to the given aspect ratio (fallback path without the editor crop)."""
    if ratio is None:
        return img
    W, H = img.size
    if W / H > ratio:
        cw = int(round(H * ratio))
        ch = H
    else:
        cw = W
        ch = int(round(W / ratio))
    cw = max(1, min(cw, W))
    ch = max(1, min(ch, H))
    l = (W - cw) // 2
    t = (H - ch) // 2
    return img.crop((l, t, l + cw, t + ch))


def _apply_target_resize(
    img: Image.Image, tw: float, th: float, fit: str, high_q: bool
) -> Image.Image:
    tw_i = max(1, int(round(float(tw or 0))))
    th_i = max(1, int(round(float(th or 0))))
    if tw_i <= 1 and th_i <= 1:
        return img
    if tw_i <= 1:
        tw_i = max(1, int(round(img.width * th_i / img.height)))
    if th_i <= 1:
        th_i = max(1, int(round(img.height * tw_i / img.width)))
    res = Image.LANCZOS if high_q else Image.BILINEAR
    if fit == FIT_STRETCH:
        return img.resize((tw_i, th_i), res)
    if fit == FIT_COVER:
        scale = max(tw_i / img.width, th_i / img.height)
        nw, nh = int(round(img.width * scale)), int(round(img.height * scale))
        im2 = img.resize((nw, nh), res)
        l = (nw - tw_i) // 2
        t = (nh - th_i) // 2
        return im2.crop((l, t, l + tw_i, t + th_i))
    # Contain (pad): scale to fit, pad with white
    scale = min(tw_i / img.width, th_i / img.height)
    nw, nh = max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))
    im2 = img.resize((nw, nh), res)
    canvas = Image.new("RGB", (tw_i, th_i), (255, 255, 255))
    canvas.paste(im2, ((tw_i - nw) // 2, (th_i - nh) // 2))
    return canvas


def _save_image(
    editor_value: dict,
    output_dir: str,
    filename: str,
    out_format: str,
    flatten_bg: bool,
    tw: float,
    th: float,
    resize_fit: str,
    do_resize: bool,
    high_q: bool,
    ratio_choice: str,
) -> Tuple[str, Optional[str]]:
    """Crop + optional ratio/resize + save. Returns (status, filepath or None)."""
    img = _crop_from_editor_value(editor_value)
    if img is None:
        return "没有可保存的图片。请先上传图片并裁剪。", None
    img = _crop_by_ratio(img, _parse_ratio(ratio_choice))
    if do_resize:
        img = _apply_target_resize(img, tw, th, resize_fit, high_q)
    output_dir = (output_dir or "").strip().strip('"')
    if not output_dir:
        output_dir = os.path.join(scriptdir, "outputs", "single_crop")
    os.makedirs(output_dir, exist_ok=True)
    filename = (filename or "").strip()
    if not filename:
        filename = "cropped"
    name, _ = os.path.splitext(filename)
    ext = ".png" if out_format == "PNG" else ".jpg"
    out_path = os.path.join(output_dir, name + ext)
    counter = 1
    while os.path.exists(out_path):
        out_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
        counter += 1
    try:
        if flatten_bg:
            img = _flatten_transparency_to_white(img)
        if ext == ".jpg":
            _flatten_transparency_to_white(img).save(out_path, quality=95)
        else:
            img.save(out_path, "PNG")
    except OSError as e:
        log.warning("Single crop save failed %s: %s", out_path, e)
        return f"保存失败：{e}", None
    size_txt = f"{img.width}x{img.height}"
    msg = f"已保存 {size_txt} → {out_path}"
    log.info(msg)
    return msg, out_path


def gradio_single_crop_tab(
    headless: bool = False,
    config: Optional[KohyaSSGUIConfig] = None,
):
    cfg = config or KohyaSSGUIConfig()
    default_out = cfg.get(
        "utilities.single_crop_output", os.path.join(scriptdir, "outputs", "single_crop")
    )

    with gr.Tab("单独裁剪"):
        gr.Markdown(
            "单图交互式裁剪：上传图片，使用编辑器中的 ✂️ **裁剪工具**"
            "（或 🔍 缩放），然后点击**保存**。"
            "比例锁定会约束裁剪工具的形状；如需要，保存时可按比例居中裁剪兜底。"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                editor = gr.ImageEditor(
                    label="裁剪编辑器（✂ 裁剪 / 🔍 缩放）",
                    sources=("upload", "clipboard"),
                    transforms=("crop", "resize"),
                    crop_size="custom",
                    canvas_size=(900, 700),
                    height=620,
                    type="pil",
                    layers=False,
                    brush=False,
                    eraser=False,
                    format="png",
                )

            with gr.Column(scale=2, min_width=260):
                gr.Markdown("#### 保存")
                output_folder = gr.Textbox(
                    label="输出文件夹",
                    value=default_out or "",
                    placeholder="裁剪后的图片保存到这里",
                )
                with gr.Row():
                    out_browse = gr.Button("📂 输出", elem_classes=["tool"], visible=not headless)
                filename = gr.Textbox(
                    label="文件名（不含扩展名）",
                    value="cropped",
                    placeholder="例如：my_image",
                )
                out_format = gr.Radio(choices=["PNG", "JPG"], value="PNG", label="格式")
                flatten_bg = gr.Checkbox(
                    label="透明背景拍平为白色",
                    value=True,
                )

                with gr.Accordion("裁剪后缩放", open=False):
                    do_resize = gr.Checkbox(label="缩放输出图片", value=False)
                    with gr.Row():
                        target_w = gr.Number(value=1024, precision=0, label="宽度（px，0 = 自动）")
                        target_h = gr.Number(value=1024, precision=0, label="高度（px，0 = 自动）")
                    resize_fit = gr.Radio(
                        choices=[FIT_STRETCH, FIT_COVER, FIT_CONTAIN],
                        value=FIT_STRETCH,
                        label="填充方式",
                    )
                    high_q = gr.Checkbox(label="高质量缩放（Lanczos）", value=True)

                with gr.Accordion("比例助手（按比例居中裁剪）", open=False):
                    gr.Markdown(
                        "编辑器的裁剪工具是自由形状。"
                        "如需精确比例，可在保存时将结果居中裁剪到指定比例。"
                    )
                    ratio_choice = gr.Radio(
                        choices=RATIO_CHOICES,
                        value=RATIO_FREE,
                        label="宽高比",
                    )

                save_btn = gr.Button("保存裁剪结果", variant="primary", visible=not headless)
                status = gr.Textbox(label="状态", interactive=False, lines=2)
                result_file = gr.File(label="已保存文件", interactive=False)

        out_browse.click(fn=get_folder_path, inputs=output_folder, outputs=output_folder)

        save_btn.click(
            fn=_save_image,
            inputs=[
                editor,
                output_folder,
                filename,
                out_format,
                flatten_bg,
                target_w,
                target_h,
                resize_fit,
                do_resize,
                high_q,
                ratio_choice,
            ],
            outputs=[status, result_file],
        )
