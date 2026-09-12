"""
Batch crop / resize for training datasets — BIRME-style layout (grid + sidebar),
with optional smart focal crop (edge energy), custom % region, and PNG rename.
Inspired by https://www.birme.net/
"""
from __future__ import annotations

import os
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

    if mode == "Custom region (%)":
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
    if mode == "Custom region (%)":
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
        return "Invalid input folder."
    if not output_dir:
        return "Set output folder."
    os.makedirs(output_dir, exist_ok=True)
    paths = _list_images(input_dir)
    if not paths:
        return "No images found in input folder."
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
        extra = f" (PNG + rename as {prefix}1.png …)"
    msg = f"Saved {n_ok}/{len(paths)} image(s) to {output_dir}{extra}"
    log.info(msg)
    return msg


def _parse_select_index(idx: Any) -> int:
    if isinstance(idx, (list, tuple)):
        return int(idx[0]) if idx else 0
    try:
        return int(idx)
    except (TypeError, ValueError):
        return 0


def gradio_batch_crop_tab(
    headless: bool = False,
    config: Optional[KohyaSSGUIConfig] = None,
):
    cfg = config or KohyaSSGUIConfig()
    default_in = cfg.get("utilities.batch_crop_input", os.path.join(scriptdir, "data"))
    default_out = cfg.get("utilities.batch_crop_output", os.path.join(scriptdir, "outputs", "batch_crop"))

    with gr.Tab("Batch Crop"):
        gr.Markdown(
            "BIRME-style batch resize/crop: **thumbnail grid** + **large preview**. "
            "Set target **width / height** and **ratio**, optional **auto width/height**, "
            "**auto focal** (edge-based crop), and **no resize** (crop only). "
            "Advanced: **custom region %** for the same rectangle on every image."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                gr.Markdown("#### Image list")
                input_folder = gr.Textbox(label="Input folder", value=default_in or "", placeholder="Folder with images")
                output_folder = gr.Textbox(
                    label="Output folder (default save location)",
                    value=default_out or "",
                    placeholder="Processed images saved here",
                )
                with gr.Row():
                    in_browse = gr.Button("📂", elem_classes=["tool"], visible=not headless)
                    out_browse = gr.Button("📂 out", elem_classes=["tool"], visible=not headless)
                    scan_btn = gr.Button("Scan folder", variant="primary", visible=not headless)
                gr.Markdown("*Re-scan the folder after you add or remove files.*")

                image_paths = gr.State([])
                selected_idx = gr.State(0)

                thumb_gallery = gr.Gallery(
                    label="Thumbnails (click for large preview)",
                    columns=4,
                    rows=2,
                    height=420,
                    object_fit="contain",
                    type="filepath",
                    interactive=False,
                    show_download_button=False,
                    allow_preview=True,
                )

                gr.Markdown("#### Large preview (green = crop region)")
                preview = gr.Image(
                    label="Preview (green box = crop area)",
                    type="pil",
                    interactive=False,
                    height=380,
                )
                preview_zoom = gr.Slider(0.2, 4.0, value=1.0, step=0.05, label="Preview zoom (display only)")
                show_box = gr.Checkbox(label="Show crop box overlay", value=True)

            with gr.Column(scale=2, min_width=260):
                gr.Markdown("#### Resize / Crop")
                mode = gr.Radio(
                    choices=["Target size (BIRME)", "Custom region (%)"],
                    value="Target size (BIRME)",
                    label="Crop mode",
                )

                with gr.Row():
                    out_w = gr.Number(value=1024, precision=0, label="Width (px)")
                    auto_w = gr.Checkbox(label="Auto width", value=False)
                with gr.Row():
                    out_h = gr.Number(value=1280, precision=0, label="Height (px)")
                    auto_h = gr.Checkbox(label="Auto height", value=False)

                with gr.Row():
                    ratio_w = gr.Number(value=4, precision=0, label="Ratio W")
                    ratio_h = gr.Number(value=5, precision=0, label="Ratio H")

                high_q = gr.Checkbox(label="High-quality resize (Lanczos)", value=True)
                auto_focal = gr.Checkbox(
                    label="Auto detect focal point (edge-based, per image)",
                    value=True,
                )
                no_resize = gr.Checkbox(
                    label="Do not resize (crop to aspect only; output size varies)",
                    value=False,
                )

                gr.Markdown("##### Manual focal nudge (from computed crop)")
                shift_x = gr.Slider(-20, 20, value=0, step=0.5, label="Shift X % of image width")
                shift_y = gr.Slider(-20, 20, value=0, step=0.5, label="Shift Y % of image height")

                with gr.Accordion("Advanced: custom region (%)", open=False):
                    gr.Markdown("Same percentage crop on **every** image (legacy mode).")
                    left_pct = gr.Slider(0, 90, value=0, step=0.5, label="Left offset %")
                    top_pct = gr.Slider(0, 90, value=0, step=0.5, label="Top offset %")
                    crop_w_pct = gr.Slider(10, 100, value=100, step=0.5, label="Crop width %")
                    crop_h_pct = gr.Slider(10, 100, value=100, step=0.5, label="Crop height %")

                with gr.Accordion("Rename & PNG (same logic as train/j2p.py)", open=False):
                    rename_png = gr.Checkbox(
                        label="Rename & save as PNG (prefix + counter, e.g. Pic1.png)",
                        value=False,
                    )
                    name_prefix = gr.Textbox(
                        label="Filename prefix",
                        value="image",
                        placeholder="e.g. Pic → Pic1.png, Pic2.png …",
                    )
                    delete_source = gr.Checkbox(
                        label="Delete original files after successful save (dangerous)",
                        value=False,
                    )

                with gr.Row():
                    reset_btn = gr.Button("Reset settings", visible=not headless)
                    clear_btn = gr.Button("Clear list", visible=not headless)

        run_btn = gr.Button("Save to folder (crop all)", variant="primary", visible=not headless)
        status = gr.Textbox(label="Status", interactive=False, lines=2)

        def _mode_key(m: str) -> str:
            return "Custom region (%)" if m == "Custom region (%)" else "Target size (BIRME)"

        def scan_folder(path, m, lp, tp, wp, hp, tw, th, zoom, show_b, af, sx, sy, nr):
            path = (path or "").strip().strip('"')
            if not path or not os.path.isdir(path):
                return [], 0, [], "Invalid folder.", None
            paths = _list_images(path)
            if not paths:
                return [], 0, [], "No images found.", None
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
            return paths, 0, gallery_items, f"Found {len(paths)} image(s).", preview_img

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
            )

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
                "Target size (BIRME)",
                0,
                0,
                100,
                100,
            )

        def clear_list():
            return [], 0, [], "List cleared. Scan again to load.", None

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
            outputs=[image_paths, selected_idx, thumb_gallery, status, preview],
        )

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
            outputs=[selected_idx, preview],
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
            ],
        ).then(fn=update_preview, inputs=preview_inputs, outputs=preview)

        clear_btn.click(fn=clear_list, outputs=[image_paths, selected_idx, thumb_gallery, status, preview])

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
            mk = "Custom region (%)" if m == "Custom region (%)" else "Target size (BIRME)"
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
