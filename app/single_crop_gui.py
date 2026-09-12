"""
Single-image crop tab — batch-tab-style controls (target resolution /
custom % region, auto focal, manual shift, PNG rename) with a
click-to-move crop box locked to the target aspect ratio (no stretching):
the box always matches the target W:H, then the crop is resized to the
exact target resolution. Common resolution presets included.
"""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw

from .class_gui_config import KohyaSSGUIConfig
from .common_gui import get_folder_path, scriptdir
from .custom_logging import setup_logging
from .batch_crop_gui import (
    _apply_shift,
    _center_crop_box_pixels,
    _crop_from_percent,
    _flatten_transparency_to_white,
    _smart_crop_box,
)

log = setup_logging()

MODE_TARGET = "按目标分辨率（无拉伸）"
MODE_REGION = "自定义区域（%）"

PRESET_CUSTOM = "自定义"
DEFAULT_PRESET = "1536×1536（1:1）"
RESOLUTION_PRESETS = [
    "2560×1440（16:9）",
    "2560×1920（4:3）",
    "1920×1080（16:9）",
    "1536×1536（1:1）",
    "1280×720（16:9）",
    "1024×1024（1:1）",
    "1216×832（3:2 横）",
    "832×1216（3:2 竖）",
]

_PREVIEW_MAX_SIDE = 900
_RESULT_MAX_SIDE = 480


def _parse_preset(choice: str) -> Tuple[Optional[int], Optional[int]]:
    if not choice or choice == PRESET_CUSTOM:
        return None, None
    m = re.search(r"(\d+)\s*[×xX*]\s*(\d+)", choice)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _fit_box_size(W: int, H: int, tw: int, th: int) -> Tuple[int, int]:
    """Largest box inside the image with exactly the tw/th aspect."""
    ra = tw / th
    if W / H > ra:
        ch = H
        cw = int(round(H * ra))
    else:
        cw = W
        ch = int(round(W / ra))
    return max(1, min(cw, W)), max(1, min(ch, H))


def _target_box(
    im: Image.Image,
    tw: int,
    th: int,
    center: Optional[Tuple[int, int]],
    auto_focal: bool,
    shift_x: float,
    shift_y: float,
) -> Tuple[int, int, int, int]:
    """Aspect-locked crop box. Clicked point wins as box center; otherwise
    edge-energy focal (if enabled) or center placement."""
    W, H = im.size
    if center is not None:
        cw, ch = _fit_box_size(W, H, tw, th)
        cx, cy = center
        l = max(0, min(int(cx - cw / 2), W - cw))
        t = max(0, min(int(cy - ch / 2), H - ch))
    elif auto_focal:
        l, t, cw, ch = _smart_crop_box(im, tw, th)
    else:
        l, t, cw, ch = _center_crop_box_pixels(im, tw, th)
    l, t = _apply_shift(l, t, cw, ch, W, H, shift_x, shift_y)
    return l, t, cw, ch


def _display(im: Image.Image, max_side: int) -> Image.Image:
    s = max_side / max(1, max(im.width, im.height))
    if s >= 1.0:
        return im.copy()
    return im.resize(
        (max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS
    )


def _render(
    im: Optional[Image.Image],
    center: Optional[Tuple[int, int]],
    mode: str,
    lp: float,
    tp: float,
    wp: float,
    hp: float,
    tw: float,
    th: float,
    high_q: bool,
    auto_focal: bool,
    shift_x: float,
    shift_y: float,
    no_resize: bool,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """Annotated full image + result preview + one-line box info."""
    if im is None:
        return None, None, "请先上传或加载图片。"
    tw_i = max(1, int(round(float(tw or 1024))))
    th_i = max(1, int(round(float(th or 1024))))
    lw = max(2, int(im.width / 400))

    if mode == MODE_REGION:
        l, t, bw, bh, cropped = _crop_from_percent(im, lp, tp, wp, hp)
        box_txt = f"裁剪区域：({l}, {t}) {bw}×{bh}px"
    else:
        l, t, bw, bh = _target_box(im, tw_i, th_i, center, auto_focal, shift_x, shift_y)
        cropped = im.crop((l, t, l + bw, t + bh))
        box_txt = f"裁剪框：({l}, {t}) {bw}×{bh}px"

    vis = im.copy()
    ImageDraw.Draw(vis).rectangle(
        [l, t, l + bw - 1, t + bh - 1], outline="#00FF88", width=lw
    )

    if no_resize:
        result = cropped
        out_txt = f"输出：{bw}×{bh}（不缩放）"
    else:
        res = Image.LANCZOS if high_q else Image.BILINEAR
        result = cropped.resize((tw_i, th_i), res)
        out_txt = f"输出：{tw_i}×{th_i}"
    return _display(vis, _PREVIEW_MAX_SIDE), _display(result, _RESULT_MAX_SIDE), f"{box_txt} ｜ {out_txt}"


def _sanitize_filename(name: str) -> str:
    """Strip characters Windows forbids in file names (path separators, etc.)."""
    return "".join("_" if c in '<>:"/\\|?*' else c for c in name).strip(" .") or "cropped"


def _save_single(
    im: Optional[Image.Image],
    center: Optional[Tuple[int, int]],
    mode: str,
    lp: float,
    tp: float,
    wp: float,
    hp: float,
    tw: float,
    th: float,
    high_q: bool,
    auto_focal: bool,
    sx: float,
    sy: float,
    no_resize: bool,
    output_dir: str,
    filename: str,
    out_format: str,
    flatten_bg: bool,
    rename_png: bool,
    name_prefix: str,
    overwrite: bool,
    delete_source_flag: bool,
    source_path: str,
) -> Tuple[str, Optional[str]]:
    try:
        return _save_single_impl(
            im, center, mode, lp, tp, wp, hp, tw, th, high_q, auto_focal,
            sx, sy, no_resize, output_dir, filename, out_format, flatten_bg,
            rename_png, name_prefix, overwrite, delete_source_flag, source_path,
        )
    except Exception as e:
        # Any escapee exception shows a bare "错误" toast in the UI; report the
        # actual reason in the status line instead.
        log.exception("Single crop save failed")
        return f"保存失败：{type(e).__name__}: {e}", None


def _save_single_impl(
    im: Optional[Image.Image],
    center: Optional[Tuple[int, int]],
    mode: str,
    lp: float,
    tp: float,
    wp: float,
    hp: float,
    tw: float,
    th: float,
    high_q: bool,
    auto_focal: bool,
    sx: float,
    sy: float,
    no_resize: bool,
    output_dir: str,
    filename: str,
    out_format: str,
    flatten_bg: bool,
    rename_png: bool,
    name_prefix: str,
    overwrite: bool,
    delete_source_flag: bool,
    source_path: str,
) -> Tuple[str, Optional[str]]:
    if im is None:
        return "没有可保存的图片。请先上传或加载图片。", None
    tw_i = max(1, int(round(float(tw or 1024))))
    th_i = max(1, int(round(float(th or 1024))))

    if mode == MODE_REGION:
        _l, _t, _bw, _bh, out = _crop_from_percent(im, lp, tp, wp, hp)
        if not no_resize:
            res = Image.LANCZOS if high_q else Image.BILINEAR
            out = out.resize((tw_i, th_i), res)
    else:
        l, t, bw, bh = _target_box(im, tw_i, th_i, center, auto_focal, sx, sy)
        out = im.crop((l, t, l + bw, t + bh))
        if not no_resize:
            res = Image.LANCZOS if high_q else Image.BILINEAR
            out = out.resize((tw_i, th_i), res)

    output_dir = (output_dir or "").strip().strip('"') or os.path.join(
        scriptdir, "outputs", "single_crop"
    )
    os.makedirs(output_dir, exist_ok=True)

    if rename_png:
        prefix = _sanitize_filename((name_prefix or "").strip() or "image")
        counter = 1
        while not overwrite and os.path.exists(
            os.path.join(output_dir, f"{prefix}{counter}.png")
        ):
            counter += 1
        out_path = os.path.join(output_dir, f"{prefix}{counter}.png")
    else:
        name = (filename or "").strip()
        if not name:
            # No filename given: keep the source image's own name when known.
            src = (source_path or "").strip().strip('"')
            if src:
                name = os.path.splitext(os.path.basename(src))[0]
        name = _sanitize_filename(name)
        ext = ".png" if out_format == "PNG" else ".jpg"
        out_path = os.path.join(output_dir, name + ext)
        if not overwrite:
            counter = 1
            while os.path.exists(out_path):
                out_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
                counter += 1

    try:
        if out_path.lower().endswith(".jpg"):
            _flatten_transparency_to_white(out).save(out_path, quality=95)
        else:
            save_img = _flatten_transparency_to_white(out) if flatten_bg else out
            save_img.save(out_path, "PNG")
    except OSError as e:
        log.warning("Single crop save failed %s: %s", out_path, e)
        return f"保存失败：{e}", None

    deleted = ""
    if delete_source_flag:
        src = (source_path or "").strip().strip('"')
        if src and os.path.isfile(src) and os.path.abspath(src) != os.path.abspath(out_path):
            try:
                os.remove(src)
                deleted = "（已删除源文件）"
            except OSError as e:
                log.warning("Could not delete source %s: %s", src, e)
                deleted = f"（删除源文件失败：{e}）"
        elif not src:
            deleted = "（未填写图片路径，跳过删除源文件）"

    msg = f"已保存 {out.width}×{out.height} → {out_path}{deleted}"
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
            "单图无拉伸裁剪：**点击图片移动绿色裁剪框**（框比例锁定为目标分辨率，绝不变形），"
            "保存时先按比例裁剪、再缩放到目标分辨率。"
            "未点击时按**自动检测焦点**（或居中）放置裁剪框；滑块可微调。"
        )

        center_state = gr.State(None)

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("#### 图片（点击移动裁剪框）")
                input_image = gr.Image(
                    label="输入图片（上传 / 粘贴 / 点击定位）",
                    type="pil",
                    sources=("upload", "clipboard"),
                    height=520,
                    interactive=True,
                )
                source_path = gr.Textbox(
                    label="图片路径（可选，用于加载与删除源文件）",
                    placeholder=r"D:/pics/foo.png",
                )
                with gr.Row():
                    load_path_btn = gr.Button("从路径加载", visible=not headless)
                    reset_btn = gr.Button("重置设置", visible=not headless)

                gr.Markdown("#### 预览（绿框 = 裁剪区域）")
                box_preview = gr.Image(label="整图 + 裁剪框", interactive=False, height=340)
                result_preview = gr.Image(label="结果预览", interactive=False, height=300)
                box_info = gr.Textbox(label="裁剪信息", interactive=False, lines=1)

            with gr.Column(scale=2, min_width=260):
                gr.Markdown("#### 缩放 / 裁剪")
                mode = gr.Radio(
                    choices=[MODE_TARGET, MODE_REGION],
                    value=MODE_TARGET,
                    label="裁剪模式",
                )

                preset = gr.Dropdown(
                    choices=[PRESET_CUSTOM] + RESOLUTION_PRESETS,
                    value=DEFAULT_PRESET,
                    label="常用分辨率预设",
                )

                with gr.Row():
                    out_w = gr.Number(value=1536, precision=0, label="宽度（px）")
                    auto_w = gr.Checkbox(label="自动宽度", value=False)
                with gr.Row():
                    out_h = gr.Number(value=1536, precision=0, label="高度（px）")
                    auto_h = gr.Checkbox(label="自动高度", value=False)

                with gr.Row():
                    ratio_w = gr.Number(value=1, precision=0, label="比例 W")
                    ratio_h = gr.Number(value=1, precision=0, label="比例 H")

                high_q = gr.Checkbox(label="高质量缩放（Lanczos）", value=True)
                auto_focal = gr.Checkbox(
                    label="自动检测焦点（边缘能量分析；未点击图片时生效）",
                    value=True,
                )
                no_resize = gr.Checkbox(
                    label="不缩放（仅按比例裁剪；输出尺寸不定）",
                    value=False,
                )

                gr.Markdown("##### 手动焦点微调（基于当前裁剪框）")
                shift_x = gr.Slider(-20, 20, value=0, step=0.5, label="水平偏移 %（相对图片宽度）")
                shift_y = gr.Slider(-20, 20, value=0, step=0.5, label="垂直偏移 %（相对图片高度）")

                with gr.Accordion("进阶：自定义区域（%）", open=False):
                    gr.Markdown("对图片应用相同的百分比裁剪（旧模式）。")
                    left_pct = gr.Slider(0, 90, value=0, step=0.5, label="左侧偏移 %")
                    top_pct = gr.Slider(0, 90, value=0, step=0.5, label="顶部偏移 %")
                    crop_w_pct = gr.Slider(10, 100, value=100, step=0.5, label="裁剪宽度 %")
                    crop_h_pct = gr.Slider(10, 100, value=100, step=0.5, label="裁剪高度 %")

                with gr.Accordion("重命名与 PNG（前缀 + 序号）", open=False):
                    rename_png = gr.Checkbox(
                        label="重命名并保存为 PNG（前缀 + 序号，如 Pic1.png）",
                        value=False,
                    )
                    name_prefix = gr.Textbox(
                        label="文件名前缀",
                        value="image",
                        placeholder="如 Pic → Pic1.png、Pic2.png …",
                    )
                    delete_source = gr.Checkbox(
                        label="保存成功后删除源文件（危险；需填写图片路径）",
                        value=False,
                    )

                gr.Markdown("#### 保存")
                output_folder = gr.Textbox(
                    label="输出文件夹",
                    value=default_out or "",
                    placeholder="裁剪后的图片保存到这里",
                )
                with gr.Row():
                    out_browse = gr.Button("📂 输出", elem_classes=["tool"], visible=not headless)
                filename = gr.Textbox(
                    label="文件名（不含扩展名；留空 = cropped，或沿用图片路径中的文件名）",
                    value="",
                    placeholder="留空则用图片路径的文件名（上传的图片无路径时为 cropped）",
                )
                overwrite = gr.Checkbox(
                    label="覆盖已存在文件（勾选后同名文件直接覆盖，不再加序号）",
                    value=False,
                )
                out_format = gr.Radio(choices=["PNG", "JPG"], value="PNG", label="格式")
                flatten_bg = gr.Checkbox(label="透明背景拍平为白色", value=True)

                save_btn = gr.Button("保存裁剪结果", variant="primary", visible=not headless)
                status = gr.Textbox(label="状态", interactive=False, lines=2)
                result_file = gr.File(label="已保存文件", interactive=False)

        # --- handlers (same conventions as the batch tab) ---

        def _update_preview(im, center, m, lp_, tp_, wp_, hp_, tw_, th_, hq, af, sx, sy, nr):
            mk = MODE_REGION if m == MODE_REGION else MODE_TARGET
            return _render(
                im, center, mk, lp_, tp_, wp_, hp_,
                float(tw_ or 1024), float(th_ or 1024), hq, af, sx, sy, nr,
            )

        def on_image_select(evt: gr.SelectData, im, center, m, lp_, tp_, wp_, hp_, tw_, th_, hq, af, sx, sy, nr):
            idx = getattr(evt, "index", None)
            if im is not None and isinstance(idx, (list, tuple)) and len(idx) == 2:
                try:
                    center = (int(idx[0]), int(idx[1]))
                except (TypeError, ValueError):
                    pass
            mk = MODE_REGION if m == MODE_REGION else MODE_TARGET
            return (center,) + _render(
                im, center, mk, lp_, tp_, wp_, hp_,
                float(tw_ or 1024), float(th_ or 1024), hq, af, sx, sy, nr,
            )

        def on_image_change(im, center, m, lp_, tp_, wp_, hp_, tw_, th_, hq, af, sx, sy, nr):
            mk = MODE_REGION if m == MODE_REGION else MODE_TARGET
            return (None,) + _render(
                im, None, mk, lp_, tp_, wp_, hp_,
                float(tw_ or 1024), float(th_ or 1024), hq, af, sx, sy, nr,
            )

        def apply_preset(choice):
            w, h = _parse_preset(choice)
            if w is None or h is None:
                return gr.update(), gr.update()
            return gr.update(value=w), gr.update(value=h)

        def load_from_path(path):
            p = (path or "").strip().strip('"')
            if not p or not os.path.isfile(p):
                return None, "路径无效，未找到文件。"
            try:
                im = Image.open(p)
                im.load()
            except OSError as e:
                return None, f"打开失败：{e}"
            return im, f"已加载 {os.path.basename(p)}（{im.width}×{im.height}px）"

        def reset_settings():
            return (
                None,
                1536,
                1536,
                1,
                1,
                False,
                False,
                True,
                False,
                True,
                0,
                0,
                0,
                0,
                100,
                100,
                MODE_TARGET,
                DEFAULT_PRESET,
            )

        def recompute_w_from_h(h, rw, rh, use_auto_w: bool):
            if not use_auto_w or rh <= 0:
                return gr.update()
            return gr.update(value=int(round(float(h) * float(rw) / float(rh))))

        def recompute_h_from_w(w, rw, rh, use_auto_h: bool):
            if not use_auto_h or rw <= 0:
                return gr.update()
            return gr.update(value=int(round(float(w) * float(rh) / float(rw))))

        def ratio_to_height(w, rw, rh):
            if rw <= 0 or rh <= 0:
                return gr.update()
            return gr.update(value=int(round(float(w) * float(rh) / float(rw))))

        def toggle_auto_w(checked):
            if checked:
                return gr.update(value=False)
            return gr.update()

        def toggle_auto_h(checked):
            if checked:
                return gr.update(value=False)
            return gr.update()

        # --- events ---

        out_browse.click(fn=get_folder_path, inputs=output_folder, outputs=output_folder)
        load_path_btn.click(fn=load_from_path, inputs=source_path, outputs=[input_image, status])

        preview_inputs = [
            input_image,
            center_state,
            mode,
            left_pct,
            top_pct,
            crop_w_pct,
            crop_h_pct,
            out_w,
            out_h,
            high_q,
            auto_focal,
            shift_x,
            shift_y,
            no_resize,
        ]

        for comp in (
            mode,
            left_pct,
            top_pct,
            crop_w_pct,
            crop_h_pct,
            out_w,
            out_h,
            high_q,
            auto_focal,
            shift_x,
            shift_y,
            no_resize,
        ):
            comp.change(
                fn=_update_preview,
                inputs=preview_inputs,
                outputs=[box_preview, result_preview, box_info],
            )

        input_image.select(
            fn=on_image_select,
            inputs=preview_inputs,
            outputs=[center_state, box_preview, result_preview, box_info],
        )
        input_image.upload(
            fn=on_image_change,
            inputs=preview_inputs,
            outputs=[center_state, box_preview, result_preview, box_info],
        )
        input_image.change(
            fn=on_image_change,
            inputs=preview_inputs,
            outputs=[center_state, box_preview, result_preview, box_info],
        )
        input_image.clear(
            fn=on_image_change,
            inputs=preview_inputs,
            outputs=[center_state, box_preview, result_preview, box_info],
        )

        preset.change(fn=apply_preset, inputs=preset, outputs=[out_w, out_h])

        auto_w.change(fn=toggle_auto_w, inputs=auto_w, outputs=auto_h)
        auto_h.change(fn=toggle_auto_h, inputs=auto_h, outputs=auto_w)

        out_h.change(
            fn=recompute_w_from_h,
            inputs=[out_h, ratio_w, ratio_h, auto_w],
            outputs=out_w,
        )
        out_w.change(
            fn=recompute_h_from_w,
            inputs=[out_w, ratio_w, ratio_h, auto_h],
            outputs=out_h,
        )
        ratio_w.change(
            fn=ratio_to_height,
            inputs=[out_w, ratio_w, ratio_h],
            outputs=out_h,
        )
        ratio_h.change(
            fn=ratio_to_height,
            inputs=[out_w, ratio_w, ratio_h],
            outputs=out_h,
        )

        reset_btn.click(
            fn=reset_settings,
            outputs=[
                center_state,
                out_w,
                out_h,
                ratio_w,
                ratio_h,
                auto_w,
                auto_h,
                auto_focal,
                no_resize,
                high_q,
                shift_x,
                shift_y,
                left_pct,
                top_pct,
                crop_w_pct,
                crop_h_pct,
                mode,
                preset,
            ],
        ).then(
            fn=_update_preview,
            inputs=preview_inputs,
            outputs=[box_preview, result_preview, box_info],
        )

        save_btn.click(
            fn=_save_single,
            inputs=[
                input_image,
                center_state,
                mode,
                left_pct,
                top_pct,
                crop_w_pct,
                crop_h_pct,
                out_w,
                out_h,
                high_q,
                auto_focal,
                shift_x,
                shift_y,
                no_resize,
                output_folder,
                filename,
                out_format,
                flatten_bg,
                rename_png,
                name_prefix,
                overwrite,
                delete_source,
                source_path,
            ],
            outputs=[status, result_file],
        )
