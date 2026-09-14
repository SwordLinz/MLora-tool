"""
Unified crop / resize tab — BIRME-style layout (folder thumbnail grid + big
preview) with optional smart focal crop (edge energy), custom % region and PNG
rename, plus resolution presets, single-image upload, prev/next navigation and
per-image save. Merges the former batch-crop and single-crop tabs.
Inspired by https://www.birme.net/
"""
from __future__ import annotations

import os
import re
from typing import Any, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except ImportError:
    cv2 = None

from .class_gui_config import KohyaSSGUIConfig
from .common_gui import get_folder_path, scriptdir
from .custom_logging import setup_logging
from .gradio_paths import allow_path

log = setup_logging()

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")


def _list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(folder), key=str.lower):
        low = name.lower()
        if any(low.endswith(e) for e in IMAGE_EXTS):
            out.append(os.path.join(folder, name))
    return out


MODE_TARGET = "按目标分辨率（无拉伸）"
MODE_REGION = "自定义区域（%）"

PRESET_CUSTOM = "自定义"
DEFAULT_PRESET = PRESET_CUSTOM
RESOLUTION_PRESETS = [
    "2560×1280（2:1）",
    "2560×1440（16:9）",
    "2560×1920（4:3）",
    "1920×1080（16:9）",
    "1536×1536（1:1）",
    "1280×720（16:9）",
    "1024×1024（1:1）",
    "1216×832（3:2 横）",
    "832×1216（3:2 竖）",
]


def _parse_preset(choice: str) -> Tuple[Optional[int], Optional[int]]:
    if not choice or choice == PRESET_CUSTOM:
        return None, None
    m = re.search(r"(\d+)\s*[×xX*]\s*(\d+)", choice)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _sanitize_filename(name: str) -> str:
    """Strip characters Windows forbids in file names (path separators, etc.)."""
    return "".join("_" if c in '<>:"/\\|?*' else c for c in name).strip(" .")


def _pos_text(idx: int, paths: List[str]) -> str:
    if not paths:
        return "未加载图片"
    idx = max(0, min(int(idx), len(paths) - 1))
    return f"第 **{idx + 1} / {len(paths)}** 张：`{os.path.basename(paths[idx])}`"


def _crop_from_percent(
    im: Image.Image, lp: float, tp: float, wp: float, hp: float
) -> Tuple[int, int, int, int, Image.Image]:
    W, H = im.size
    l = int(round(W * lp / 100.0))
    t = int(round(H * tp / 100.0))
    w = int(round(W * wp / 100.0))
    h = int(round(H * hp / 100.0))
    w = max(1, min(w, W - l))
    h = max(1, min(h, H - t))
    l = max(0, min(l, W - 1))
    t = max(0, min(t, H - 1))
    if l + w > W:
        w = W - l
    if t + h > H:
        h = H - t
    cropped = im.crop((l, t, l + w, t + h))
    return l, t, w, h, cropped


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


def _save_png_j2p_style(img: Image.Image, path: str) -> None:
    out = _flatten_transparency_to_white(img)
    out.save(path, "PNG")


def _center_crop_box_pixels(im: Image.Image, tw: int, th: int) -> Tuple[int, int, int, int]:
    W, H = im.size
    if W <= 0 or H <= 0 or tw <= 0 or th <= 0:
        return 0, 0, max(1, W), max(1, H)
    ra = tw / th
    if W / H > ra:
        ch = H
        cw = int(round(H * ra))
    else:
        cw = W
        ch = int(round(W / ra))
    cw = max(1, min(cw, W))
    ch = max(1, min(ch, H))
    l = (W - cw) // 2
    t = (H - ch) // 2
    return l, t, cw, ch


def _smart_crop_box(im: Image.Image, tw: int, th: int) -> Tuple[int, int, int, int]:
    """Pick crop window with target aspect tw/th by maximizing edge energy (BIRME-like focal)."""
    if cv2 is None:
        return _center_crop_box_pixels(im, tw, th)
    W, H = im.size
    if W <= 0 or H <= 0 or tw <= 0 or th <= 0:
        return _center_crop_box_pixels(im, tw, th)
    ra = tw / th
    if W / H > ra:
        ch = H
        cw = int(round(H * ra))
    else:
        cw = W
        ch = int(round(W / ra))
    cw = max(1, min(cw, W))
    ch = max(1, min(ch, H))
    max_dim = 256
    if W >= H:
        sw = max_dim
        sh = max(1, int(round(max_dim * H / W)))
    else:
        sh = max_dim
        sw = max(1, int(round(max_dim * W / H)))
    small = im.resize((sw, sh), Image.LANCZOS).convert("L")
    arr = np.asarray(small, dtype=np.float32)
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1)
    energy = gx * gx + gy * gy
    win_w = max(1, int(round(cw * sw / W)))
    win_h = max(1, int(round(ch * sh / H)))
    win_w = min(win_w, sw)
    win_h = min(win_h, sh)
    if win_w > sw or win_h > sh:
        return _center_crop_box_pixels(im, tw, th)
    best_x, best_y, best_s = 0, 0, -1.0
    step = max(1, min(win_w, win_h) // 12)
    for y in range(0, sh - win_h + 1, step):
        for x in range(0, sw - win_w + 1, step):
            s = float(energy[y : y + win_h, x : x + win_w].sum())
            if s > best_s:
                best_s = s
                best_x, best_y = x, y
    l = int(round(best_x * W / sw))
    t = int(round(best_y * H / sh))
    l = max(0, min(l, W - cw))
    t = max(0, min(t, H - ch))
    return l, t, cw, ch


def _apply_shift(
    l: int, t: int, cw: int, ch: int, W: int, H: int, px: float, py: float
) -> Tuple[int, int]:
    nl = l + int(px * W / 100.0)
    nt = t + int(py * H / 100.0)
    nl = max(0, min(nl, W - cw))
    nt = max(0, min(nt, H - ch))
    return nl, nt


def _preview_with_box(
    path: str,
    mode: str,
    lp: float,
    tp: float,
    wp: float,
    hp: float,
    tw: float,
    th: float,
    zoom: float,
    show_box: bool,
    auto_focal: bool,
    shift_x: float,
    shift_y: float,
    no_resize: bool,
) -> Optional[Image.Image]:
    if not path or not os.path.isfile(path):
        return None
    try:
        im = Image.open(path).convert("RGB")
    except OSError as e:
        log.warning("Preview open failed %s: %s", path, e)
        return None
    zoom = max(0.1, min(float(zoom), 8.0))
    lw = max(2, int(im.width / 400))
    tw_i = max(1, int(round(float(tw or 1024))))
    th_i = max(1, int(round(float(th or 1024))))

    if mode == MODE_REGION:
        _l, _t, _w, _h, _c = _crop_from_percent(im, lp, tp, wp, hp)
        vis = im.copy()
        if show_box:
            draw = ImageDraw.Draw(vis)
            draw.rectangle([_l, _t, _l + _w - 1, _t + _h - 1], outline="#00FF88", width=lw)
        if no_resize:
            out = vis
        else:
            # Crop from vis so the green outline is preserved in the preview
            tmp = vis.crop((_l, _t, _l + _w, _t + _h)).resize((tw_i, th_i), Image.LANCZOS)
            out = tmp
    else:
        W, H = im.size
        if auto_focal:
            l, t, cw, ch = _smart_crop_box(im, tw_i, th_i)
        else:
            l, t, cw, ch = _center_crop_box_pixels(im, tw_i, th_i)
        l, t = _apply_shift(l, t, cw, ch, W, H, shift_x, shift_y)
        vis = im.copy()
        if show_box:
            draw = ImageDraw.Draw(vis)
            draw.rectangle([l, t, l + cw - 1, t + ch - 1], outline="#00FF88", width=lw)
        if no_resize:
            out = vis.crop((l, t, l + cw, t + ch))
        else:
            # Crop from vis (with overlay), not from im — otherwise the green box vanishes
            cropped = vis.crop((l, t, l + cw, t + ch))
            out = cropped.resize((tw_i, th_i), Image.LANCZOS)

    nw = max(1, int(out.width * zoom))
    nh = max(1, int(out.height * zoom))
    return out.resize((nw, nh), Image.LANCZOS)


def _process_one_image(
    im: Image.Image,
    mode: str,
    lp: float,
    tp: float,
    wp: float,
    hp: float,
    tw_i: int,
    th_i: int,
    high_q: bool,
    auto_focal: bool,
    shift_x: float,
    shift_y: float,
    no_resize: bool,
) -> Image.Image:
    if mode == MODE_REGION:
        _l, _t, _w, _h, cropped = _crop_from_percent(im, lp, tp, wp, hp)
        if no_resize:
            return cropped
        res = Image.LANCZOS if high_q else Image.BILINEAR
        return cropped.resize((tw_i, th_i), res)

    W, H = im.size
    if auto_focal:
        l, t, cw, ch = _smart_crop_box(im, tw_i, th_i)
    else:
        l, t, cw, ch = _center_crop_box_pixels(im, tw_i, th_i)
    l, t = _apply_shift(l, t, cw, ch, W, H, shift_x, shift_y)
    cropped = im.crop((l, t, l + cw, t + ch))
    if no_resize:
        return cropped
    res = Image.LANCZOS if high_q else Image.BILINEAR
    return cropped.resize((tw_i, th_i), res)


def _process_all(
    input_dir: str,
    output_dir: str,
    mode: str,
    lp: float,
    tp: float,
    wp: float,
    hp: float,
    tw: float,
    th: float,
    high_q: bool,
    rename_png: bool,
    name_prefix: str,
    delete_source: bool,
    auto_focal: bool,
    shift_x: float,
    shift_y: float,
    no_resize: bool,
) -> str:
    input_dir = (input_dir or "").strip().strip('"')
    output_dir = (output_dir or "").strip().strip('"')
    if not input_dir or not os.path.isdir(input_dir):
        return "输入文件夹无效。"
    if not output_dir:
        return "请设置输出文件夹。"
    os.makedirs(output_dir, exist_ok=True)
    paths = _list_images(input_dir)
    if not paths:
        return "输入文件夹中未找到图片。"
    tw_i = max(1, int(round(float(tw or 1024))))
    th_i = max(1, int(round(float(th or 1024))))
    prefix = (name_prefix or "").strip() or "image"
    n_ok = 0
    counter = 1
    for path in paths:
        try:
            im = Image.open(path)
            im.load()
        except OSError:
            continue
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        ext = ext.lower() if ext else ".png"
        if ext not in IMAGE_EXTS:
            ext = ".png"
        try:
            out = _process_one_image(
                im,
                mode,
                lp,
                tp,
                wp,
                hp,
                tw_i,
                th_i,
                high_q,
                auto_focal,
                shift_x,
                shift_y,
                no_resize,
            )

            if rename_png:
                out_path = os.path.join(output_dir, f"{prefix}{counter}.png")
                if os.path.abspath(path) == os.path.abspath(out_path):
                    counter += 1
                    continue
                _save_png_j2p_style(out, out_path)
                n_ok += 1
                if delete_source and os.path.isfile(path):
                    try:
                        os.remove(path)
                    except OSError as e:
                        log.warning("Could not delete source %s: %s", path, e)
                counter += 1
            else:
                out_path = os.path.join(output_dir, name + ext)
                flat = _flatten_transparency_to_white(out)
                if ext in (".jpg", ".jpeg", ".webp"):
                    flat.save(out_path, quality=95)
                else:
                    flat.save(out_path)
                n_ok += 1
        except Exception as e:
            log.warning("Save failed %s: %s", path, e)
    extra = ""
    if rename_png:
        extra = f"（PNG 重命名为 {prefix}1.png …）"
    msg = f"已保存 {n_ok}/{len(paths)} 张图片到 {output_dir}{extra}"
    log.info(msg)
    return msg


def _parse_select_index(idx: Any) -> int:
    if isinstance(idx, (list, tuple)):
        return int(idx[0]) if idx else 0
    try:
        return int(idx)
    except (TypeError, ValueError):
        return 0


def gradio_crop_tab(
    headless: bool = False,
    config: Optional[KohyaSSGUIConfig] = None,
):
    cfg = config or KohyaSSGUIConfig()
    default_in = cfg.get("utilities.batch_crop_input", os.path.join(scriptdir, "data"))
    default_out = cfg.get("utilities.batch_crop_output", os.path.join(scriptdir, "outputs", "batch_crop"))

    with gr.Tab("裁剪"):
        gr.Markdown(
            "**① 载入**：扫描**文件夹**得到缩略图网格，或用上方**上传 / 粘贴**载入单张图片。"
            "**② 调参**：在右侧设置目标**宽 / 高**或选**分辨率预设**，可选**自动宽/高**、**自动焦点**（边缘能量）与**不缩放**（仅裁剪）。"
            "**③ 保存**：**保存并下一张 ▶** 会存好当前这张并自动跳到下一张（最顺手）；"
            "**保存当前图片**只存这一张；**保存到文件夹**则整批处理。"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("#### 单张图片（上传 / 粘贴）")
                single_upload = gr.Image(
                    label="拖入 / 粘贴单张图片（会替换下方列表）",
                    type="filepath",
                    sources=("upload", "clipboard"),
                    interactive=True,
                    height=180,
                )

                gr.Markdown("#### 图片列表（文件夹）")
                input_folder = gr.Textbox(label="输入文件夹", value=default_in or "", placeholder="包含图片的文件夹")
                output_folder = gr.Textbox(
                    label="输出文件夹（默认保存位置）",
                    value=default_out or "",
                    placeholder="处理后的图片保存到这里",
                )
                with gr.Row():
                    in_browse = gr.Button("📂", elem_classes=["tool"], visible=not headless)
                    out_browse = gr.Button("📂 输出", elem_classes=["tool"], visible=not headless)
                    scan_btn = gr.Button("扫描文件夹", variant="primary", visible=not headless)
                gr.Markdown("*添加或删除文件后请重新扫描。*")

                image_paths = gr.State([])
                selected_idx = gr.State(0)

                thumb_gallery = gr.Gallery(
                    label="缩略图（点击查看大图预览）",
                    columns=4,
                    rows=2,
                    height=420,
                    object_fit="contain",
                    type="filepath",
                    interactive=False,
                    show_download_button=False,
                    allow_preview=True,
                )
                with gr.Row():
                    prev_btn = gr.Button("◀ 上一张", visible=not headless)
                    pos_label = gr.Markdown("未加载图片")
                    next_btn = gr.Button("下一张 ▶", visible=not headless)

                gr.Markdown("#### 大图预览（绿框 = 裁剪区域）")
                preview = gr.Image(
                    label="预览（绿框 = 裁剪区域）",
                    type="pil",
                    interactive=False,
                    height=380,
                )
                preview_zoom = gr.Slider(0.2, 4.0, value=1.0, step=0.05, label="预览缩放（仅显示）")
                show_box = gr.Checkbox(label="显示裁剪框叠加", value=True)

            with gr.Column(scale=2, min_width=320):
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
                    out_w = gr.Number(value=1024, precision=0, label="宽度（px）")
                    auto_w = gr.Checkbox(label="自动宽度", value=False)
                with gr.Row():
                    out_h = gr.Number(value=1280, precision=0, label="高度（px）")
                    auto_h = gr.Checkbox(label="自动高度", value=False)

                swap_wh_btn = gr.Button("⇄ 宽高互换（W ↔ H）")

                with gr.Row():
                    ratio_w = gr.Number(value=4, precision=0, label="比例 W")
                    ratio_h = gr.Number(value=5, precision=0, label="比例 H")

                high_q = gr.Checkbox(label="高质量缩放（Lanczos）", value=True)
                auto_focal = gr.Checkbox(
                    label="自动检测焦点（逐张边缘能量分析）",
                    value=True,
                )
                no_resize = gr.Checkbox(
                    label="不缩放（仅按比例裁剪；输出尺寸不定）",
                    value=False,
                )

                gr.Markdown("##### 手动焦点微调（基于计算的裁剪框）")
                shift_x = gr.Slider(-20, 20, value=0, step=0.5, label="水平偏移 %（相对图片宽度）")
                shift_y = gr.Slider(-20, 20, value=0, step=0.5, label="垂直偏移 %（相对图片高度）")

                with gr.Group(elem_classes=["sd-save-card"]):
                    gr.Markdown(
                        "#### 💾 保存当前图片\n"
                        "改完参数直接保存；点 **保存并下一张 ▶** 会存好当前这张并自动跳到下一张。"
                    )
                    cur_filename = gr.Textbox(
                        label="文件名（不含扩展名；留空 = 沿用原文件名）",
                        value="",
                        placeholder="留空则用当前图片的文件名",
                    )
                    cur_format = gr.Radio(
                        choices=["PNG", "JPG"], value="PNG", label="格式"
                    )
                    cur_overwrite = gr.Checkbox(
                        label="覆盖同名文件（不勾选则自动加序号）",
                        value=False,
                    )
                    with gr.Row():
                        save_cur_btn = gr.Button(
                            "保存当前图片",
                            variant="primary",
                            visible=not headless,
                            scale=1,
                        )
                        save_next_btn = gr.Button(
                            "保存并下一张 ▶",
                            variant="secondary",
                            visible=not headless,
                            scale=1,
                        )
                    save_status = gr.Markdown("", elem_classes=["sd-save-status"])
                    result_file = gr.File(
                        label="最近保存的文件",
                        interactive=False,
                        height=80,
                        visible=False,
                        elem_classes=["sd-save-file"],
                    )

                with gr.Accordion("进阶：自定义区域（%）", open=False):
                    gr.Markdown("对**每张**图片应用相同的百分比裁剪（旧模式）。")
                    left_pct = gr.Slider(0, 90, value=0, step=0.5, label="左侧偏移 %")
                    top_pct = gr.Slider(0, 90, value=0, step=0.5, label="顶部偏移 %")
                    crop_w_pct = gr.Slider(10, 100, value=100, step=0.5, label="裁剪宽度 %")
                    crop_h_pct = gr.Slider(10, 100, value=100, step=0.5, label="裁剪高度 %")

                with gr.Accordion("重命名与 PNG（同 train/j2p.py 逻辑）", open=False):
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
                        label="保存成功后删除原文件（危险）",
                        value=False,
                    )

                with gr.Row():
                    reset_btn = gr.Button("重置设置", visible=not headless)
                    clear_btn = gr.Button("清空列表", visible=not headless)

        run_btn = gr.Button("保存到文件夹（裁剪全部）", variant="primary", visible=not headless)
        status = gr.Textbox(label="状态", interactive=False, lines=2)

        def _mode_key(m: str) -> str:
            return MODE_REGION if m == MODE_REGION else MODE_TARGET

        def scan_folder(path, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr):
            path = (path or "").strip().strip('"')
            if not path or not os.path.isdir(path):
                return [], 0, [], "文件夹无效。", None, "未加载图片"
            paths = _list_images(path)
            if not paths:
                return [], 0, [], "未找到图片。", None, "未加载图片"
            allow_path(path)
            gallery_items = [(p, os.path.basename(p)) for p in paths]
            mk = _mode_key(m)
            preview_img = _preview_with_box(
                paths[0],
                mk,
                lp,
                tp,
                wp,
                hp,
                float(tw or 1024),
                float(th or 1280),
                zoom,
                show_b,
                af,
                sx,
                sy,
                nr,
            )
            return (
                paths,
                0,
                gallery_items,
                f"找到 {len(paths)} 张图片。",
                preview_img,
                _pos_text(0, paths),
            )

        def _path_at(paths: List[str], idx: int) -> str:
            if not paths:
                return ""
            i = max(0, min(int(idx), len(paths) - 1))
            return paths[i]

        def update_preview(paths, idx, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr):
            p = _path_at(paths or [], _parse_select_index(idx))
            mk = _mode_key(m)
            return _preview_with_box(
                p,
                mk,
                lp,
                tp,
                wp,
                hp,
                float(tw or 1024),
                float(th or 1280),
                zoom,
                show_b,
                af,
                sx,
                sy,
                nr,
            )

        def on_gallery_select(evt: gr.SelectData, paths, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr):
            idx = _parse_select_index(evt.index)
            p = _path_at(paths or [], idx)
            mk = _mode_key(m)
            return (
                idx,
                _preview_with_box(
                    p,
                    mk,
                    lp,
                    tp,
                    wp,
                    hp,
                    float(tw or 1024),
                    float(th or 1280),
                    zoom,
                    show_b,
                    af,
                    sx,
                    sy,
                    nr,
                ),
                _pos_text(idx, paths or []),
            )

        def step_image(paths, idx, delta, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr):
            """Cycle through the loaded list and refresh the preview."""
            if not paths:
                return 0, None, "请先扫描文件夹或上传单张图片。", _pos_text(0, paths or [])
            nidx = (int(idx or 0) + delta) % len(paths)
            mk = _mode_key(m)
            preview_img = _preview_with_box(
                _path_at(paths, nidx),
                mk,
                lp,
                tp,
                wp,
                hp,
                float(tw or 1024),
                float(th or 1280),
                zoom,
                show_b,
                af,
                sx,
                sy,
                nr,
            )
            return nidx, preview_img, "", _pos_text(nidx, paths)

        def go_prev(paths, idx, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr):
            return step_image(paths, idx, -1, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr)

        def go_next(paths, idx, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr):
            return step_image(paths, idx, 1, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr)

        def load_single_image(path, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr):
            """Turn an uploaded/pasted image into a one-item list and preview it."""
            p = (path or "").strip().strip('"') if isinstance(path, str) else ""
            if not p or not os.path.isfile(p):
                # Cleared/invalid upload: keep whatever list is already loaded.
                return (gr.update(),) * 6
            mk = _mode_key(m)
            preview_img = _preview_with_box(
                p,
                mk,
                lp,
                tp,
                wp,
                hp,
                float(tw or 1024),
                float(th or 1280),
                zoom,
                show_b,
                af,
                sx,
                sy,
                nr,
            )
            return [p], 0, [(p, os.path.basename(p))], "已载入单张图片。", preview_img, _pos_text(0, [p])

        def reset_settings():
            return (
                1024,
                1280,
                4,
                5,
                False,
                False,
                True,
                False,
                True,
                0,
                0,
                1.0,
                True,
                MODE_TARGET,
                0,
                0,
                100,
                100,
                PRESET_CUSTOM,
            )

        def clear_list():
            return [], 0, [], "列表已清空，请重新扫描加载。", None, "未加载图片"

        def apply_preset(choice):
            w, h = _parse_preset(choice)
            if w is None or h is None:
                return gr.update(), gr.update()
            return gr.update(value=w), gr.update(value=h)

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

        def swap_wh(w, h):
            return gr.update(value=h), gr.update(value=w)

        def save_current(
            paths,
            idx,
            m,
            lp,
            tp,
            wp,
            hp,
            tw,
            th,
            hq,
            af,
            sx,
            sy,
            nr,
            out_dir,
            filename,
            out_format,
            overwrite,
        ):
            p = _path_at(paths or [], _parse_select_index(idx))
            if not p or not os.path.isfile(p):
                return "没有可保存的图片，请先扫描文件夹或上传单张图片。", None
            out_dir = (out_dir or "").strip().strip('"') or os.path.join(
                scriptdir, "outputs", "crop"
            )
            os.makedirs(out_dir, exist_ok=True)
            allow_path(out_dir)
            tw_i = max(1, int(round(float(tw or 1024))))
            th_i = max(1, int(round(float(th or 1280))))
            try:
                im = Image.open(p)
                im.load()
                im = im.convert("RGB")
            except OSError as e:
                log.warning("Single save open failed %s: %s", p, e)
                return f"打开失败：{e}", None
            try:
                out = _process_one_image(
                    im,
                    _mode_key(m),
                    lp,
                    tp,
                    wp,
                    hp,
                    tw_i,
                    th_i,
                    hq,
                    af,
                    sx,
                    sy,
                    nr,
                )
                name = _sanitize_filename((filename or "").strip())
                if not name:
                    name = _sanitize_filename(os.path.splitext(os.path.basename(p))[0])
                name = name or "cropped"
                ext = ".png" if out_format == "PNG" else ".jpg"
                out_path = os.path.join(out_dir, name + ext)
                if not overwrite:
                    counter = 1
                    while os.path.exists(out_path):
                        out_path = os.path.join(out_dir, f"{name}_{counter}{ext}")
                        counter += 1
                if ext == ".jpg":
                    _flatten_transparency_to_white(out).save(out_path, quality=95)
                else:
                    _flatten_transparency_to_white(out).save(out_path, "PNG")
            except Exception as e:
                log.exception("Single save failed")
                return f"保存失败：{type(e).__name__}: {e}", None
            msg = f"已保存 {out.width}×{out.height}：`{os.path.basename(out_path)}`"
            log.info("已保存 %s", out_path)
            return msg, out_path

        def save_and_next(
            paths,
            idx,
            m,
            lp,
            tp,
            wp,
            hp,
            tw,
            th,
            hq,
            af,
            sx,
            sy,
            nr,
            out_dir,
            filename,
            out_format,
            overwrite,
            zoom,
            show_b,
        ):
            """Save the current image, then advance to the next one in the list."""
            msg, out_path = save_current(
                paths,
                idx,
                m,
                lp,
                tp,
                wp,
                hp,
                tw,
                th,
                hq,
                af,
                sx,
                sy,
                nr,
                out_dir,
                filename,
                out_format,
                overwrite,
            )
            cur = _parse_select_index(idx)
            if not paths or out_path is None:
                return msg, None, cur, gr.update(), _pos_text(cur, paths or [])
            nidx = (cur + 1) % len(paths)
            preview_img = _preview_with_box(
                _path_at(paths, nidx),
                _mode_key(m),
                lp,
                tp,
                wp,
                hp,
                float(tw or 1024),
                float(th or 1280),
                float(zoom or 1.0),
                show_b,
                af,
                sx,
                sy,
                nr,
            )
            return (
                f"{msg}\n\n▶ 已跳到第 **{nidx + 1} / {len(paths)}** 张。",
                out_path,
                nidx,
                preview_img,
                _pos_text(nidx, paths),
            )

        in_browse.click(fn=get_folder_path, inputs=input_folder, outputs=input_folder)
        out_browse.click(fn=get_folder_path, inputs=output_folder, outputs=output_folder)
        scan_btn.click(
            fn=scan_folder,
            inputs=[
                input_folder,
                mode,
                left_pct,
                top_pct,
                crop_w_pct,
                crop_h_pct,
                out_w,
                out_h,
                preview_zoom,
                show_box,
                auto_focal,
                shift_x,
                shift_y,
                no_resize,
            ],
            outputs=[image_paths, selected_idx, thumb_gallery, status, preview, pos_label],
        )

        swap_wh_btn.click(fn=swap_wh, inputs=[out_w, out_h], outputs=[out_w, out_h])

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

        preview_inputs = [
            image_paths,
            selected_idx,
            mode,
            left_pct,
            top_pct,
            crop_w_pct,
            crop_h_pct,
            out_w,
            out_h,
            preview_zoom,
            show_box,
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
            preview_zoom,
            show_box,
            auto_focal,
            shift_x,
            shift_y,
            no_resize,
        ):
            comp.change(fn=update_preview, inputs=preview_inputs, outputs=preview)

        thumb_gallery.select(
            fn=on_gallery_select,
            inputs=[
                image_paths,
                mode,
                left_pct,
                top_pct,
                crop_w_pct,
                crop_h_pct,
                out_w,
                out_h,
                preview_zoom,
                show_box,
                auto_focal,
                shift_x,
                shift_y,
                no_resize,
            ],
            outputs=[selected_idx, preview, pos_label],
        )

        nav_inputs = preview_inputs
        nav_outputs = [selected_idx, preview, status, pos_label]
        prev_btn.click(fn=go_prev, inputs=nav_inputs, outputs=nav_outputs)
        next_btn.click(fn=go_next, inputs=nav_inputs, outputs=nav_outputs)

        single_inputs = [single_upload] + preview_inputs[2:]
        single_upload.change(
            fn=load_single_image,
            inputs=single_inputs,
            outputs=[image_paths, selected_idx, thumb_gallery, status, preview, pos_label],
        )

        save_inputs = [
            image_paths,
            selected_idx,
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
            cur_filename,
            cur_format,
            cur_overwrite,
        ]

        def _save_current_ui(*args):
            msg, out_path = save_current(*args)
            if not out_path:
                return msg, gr.update()
            return msg, gr.update(value=out_path, visible=True)

        def _save_and_next_ui(*args):
            msg, out_path, nidx, preview_img, pos = save_and_next(*args)
            if not out_path:
                return msg, gr.update(), nidx, preview_img, pos
            return (
                msg,
                gr.update(value=out_path, visible=True),
                nidx,
                preview_img,
                pos,
            )

        save_cur_btn.click(
            fn=_save_current_ui,
            inputs=save_inputs,
            outputs=[save_status, result_file],
        )
        save_next_btn.click(
            fn=_save_and_next_ui,
            inputs=save_inputs + [preview_zoom, show_box],
            outputs=[save_status, result_file, selected_idx, preview, pos_label],
        )

        reset_btn.click(
            fn=reset_settings,
            outputs=[
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
                preview_zoom,
                show_box,
                mode,
                left_pct,
                top_pct,
                crop_w_pct,
                crop_h_pct,
                preset,
            ],
        ).then(fn=update_preview, inputs=preview_inputs, outputs=preview)

        clear_btn.click(
            fn=clear_list,
            outputs=[image_paths, selected_idx, thumb_gallery, status, preview, pos_label],
        )

        def _run_wrapper(
            in_dir,
            out_dir,
            m,
            lp,
            tp,
            wp,
            hp,
            tw,
            th,
            hq,
            rpng,
            prefix,
            del_src,
            af,
            sx,
            sy,
            nr,
        ):
            mk = MODE_REGION if m == MODE_REGION else MODE_TARGET
            return _process_all(
                in_dir,
                out_dir,
                mk,
                lp,
                tp,
                wp,
                hp,
                tw,
                th,
                hq,
                rpng,
                prefix,
                del_src,
                af,
                sx,
                sy,
                nr,
            )

        run_btn.click(
            fn=_run_wrapper,
            inputs=[
                input_folder,
                output_folder,
                mode,
                left_pct,
                top_pct,
                crop_w_pct,
                crop_h_pct,
                out_w,
                out_h,
                high_q,
                rename_png,
                name_prefix,
                delete_source,
                auto_focal,
                shift_x,
                shift_y,
                no_resize,
            ],
            outputs=status,
        )
