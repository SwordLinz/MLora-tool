"""
Dataset tag manager (BDTM-like): three-pane UI for image folder + per-image txt tags + global tag stats.
Supports CRUD, search/filter, and optional translation via deep-translator.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, List, Optional, Tuple

import gradio as gr

from .class_gui_config import KohyaSSGUIConfig
from .common_gui import get_folder_path, scriptdir
from .custom_logging import setup_logging

log = setup_logging()

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

try:
    from deep_translator import GoogleTranslator

    _HAS_TRANSLATOR = True
except ImportError:
    GoogleTranslator = None  # type: ignore
    _HAS_TRANSLATOR = False


def _parse_tags(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if "," in text:
        return [t.strip() for t in text.split(",") if t.strip()]
    return [t.strip() for t in text.splitlines() if t.strip()]


def _parse_batch_tags(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    tags: List[str] = []
    for line in text.splitlines():
        for part in line.split(","):
            tag = part.strip()
            if tag and tag not in tags:
                tags.append(tag)
    return tags


def _join_tags(tags: List[str]) -> str:
    return ", ".join(tags)


def _caption_path(image_path: str, ext: str) -> str:
    base = os.path.splitext(image_path)[0]
    return base + ext


def _list_images(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(folder), key=str.lower):
        low = name.lower()
        if any(low.endswith(e) for e in IMAGE_EXTS):
            out.append(os.path.join(folder, name))
    return out


def _read_tags_file(txt_path: str) -> str:
    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        log.warning("Read failed %s: %s", txt_path, e)
        return ""


def _write_tags_file(txt_path: str, content: str) -> None:
    parent = os.path.dirname(os.path.abspath(txt_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content.strip() + ("\n" if content.strip() else ""))


def _global_tag_counts(folder: str, caption_ext: str) -> List[Tuple[str, int]]:
    if not folder or not os.path.isdir(folder):
        return []
    counts: Counter[str] = Counter()
    for name in os.listdir(folder):
        if not name.lower().endswith(caption_ext.lower()):
            continue
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        text = _read_tags_file(path)
        for t in _parse_tags(text):
            counts[t] += 1
    return sorted(counts.items(), key=lambda x: (-x[1], x[0].lower()))


def _list_caption_files(folder: str, ext: str) -> List[str]:
    """All caption files in folder matching extension (not only those paired with images)."""
    if not folder or not os.path.isdir(folder):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(folder), key=str.lower):
        if not name.lower().endswith(ext.lower()):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            out.append(path)
    return out


def _rows_from_counts(pairs: List[Tuple[str, int]], filter_text: str) -> List[List[Any]]:
    ft = (filter_text or "").strip().lower()
    rows: List[List[Any]] = []
    for tag, c in pairs:
        if ft and ft not in tag.lower():
            continue
        rows.append([tag, c])
    return rows


def _translate_block(text: str, target: str) -> Tuple[str, str]:
    if not _HAS_TRANSLATOR or not GoogleTranslator:
        return text, "Install: pip install deep-translator"
    text = text or ""
    if not text.strip():
        return text, ""
    parts = _parse_tags(text)
    if not parts:
        return text, ""
    out: List[str] = []
    err = ""
    try:
        translator = GoogleTranslator(source="auto", target=target)
        for p in parts:
            try:
                tr = translator.translate(p)
                out.append(tr.strip() if tr else p)
            except Exception as e:
                out.append(p)
                err = str(e)
    except Exception as e:
        return text, str(e)
    return ", ".join(out), err or ""


def _batch_add_tag_apply(
    root: str,
    ext: str,
    tag: str,
    position_label: str,
    skip_existing: bool,
) -> Tuple[str, List[Tuple[str, int]], List[List[Any]]]:
    """Insert one tag at the beginning or end of every caption file in folder."""
    root = (root or "").strip().strip('"')
    tag = (tag or "").strip()
    if not root or not os.path.isdir(root):
        return "Invalid folder.", [], []
    if not tag:
        return "Enter a tag to add.", [], []
    pos = (position_label or "").lower()
    top = pos.startswith("top") or "beginning" in pos or "start" in pos or "front" in pos
    paths = _list_caption_files(root, ext)
    if not paths:
        return f"No *{ext} caption files in folder.", [], []
    changed = 0
    skipped = 0
    for path in paths:
        raw = _read_tags_file(path)
        tags = _parse_tags(raw)
        if skip_existing and tag in tags:
            skipped += 1
            continue
        new_tags = [tag] + tags if top else tags + [tag]
        _write_tags_file(path, _join_tags(new_tags))
        changed += 1
    pairs = _global_tag_counts(root, ext)
    rows = _rows_from_counts(pairs, "")
    msg = (
        f"Batch add: {changed} file(s) updated, {skipped} skipped (tag already present). "
        f"Position: {'beginning' if top else 'end'}."
    )
    log.info(msg)
    return msg, pairs, rows


def _batch_remove_tags_apply(
    root: str,
    ext: str,
    tags_text: str,
) -> Tuple[str, List[Tuple[str, int]], List[List[Any]]]:
    """Remove one or more exact-match tags from every caption file in folder."""
    root = (root or "").strip().strip('"')
    tags_to_remove = set(_parse_batch_tags(tags_text))
    if not root or not os.path.isdir(root):
        return "Invalid folder.", [], []
    if not tags_to_remove:
        return "Enter one or more tags to remove.", [], []
    paths = _list_caption_files(root, ext)
    if not paths:
        return f"No *{ext} caption files in folder.", [], []

    changed = 0
    removed = 0
    for path in paths:
        raw = _read_tags_file(path)
        tags = _parse_tags(raw)
        new_tags = [t for t in tags if t not in tags_to_remove]
        removed_here = len(tags) - len(new_tags)
        if not removed_here:
            continue
        _write_tags_file(path, _join_tags(new_tags))
        changed += 1
        removed += removed_here

    pairs = _global_tag_counts(root, ext)
    rows = _rows_from_counts(pairs, "")
    msg = (
        f"Batch remove: {removed} tag occurrence(s) removed from {changed} file(s). "
        f"Tags: {', '.join(sorted(tags_to_remove, key=str.lower))}."
    )
    log.info(msg)
    return msg, pairs, rows


def gradio_dataset_tag_manager_tab(
    headless: bool = False,
    config: Optional[KohyaSSGUIConfig] = None,
):
    cfg = config or KohyaSSGUIConfig()
    default_dir = cfg.get("utilities.dataset_tag_manager_dir", os.path.join(scriptdir, "data"))

    with gr.Tab("Dataset Tag Manager"):
        gr.Markdown(
            "Browse a folder of images with matching caption files (`.txt` / `.caption`). "
            "Left: thumbnails; center: tags for the selected image; right: tag frequency across all caption files. "
            "Tags are saved as comma-separated text (kohya / danbooru style)."
        )

        folder = gr.Textbox(
            label="Dataset folder",
            value=default_dir or "",
            placeholder="Folder containing images and caption files",
        )
        caption_ext = gr.Dropdown(
            choices=[".txt", ".caption"],
            value=".txt",
            label="Caption extension",
        )
        with gr.Row():
            browse = gr.Button("📂", elem_classes=["tool"], visible=not headless)
            refresh = gr.Button("Refresh", visible=not headless)
        status = gr.Textbox(label="Status", lines=1, interactive=False)

        image_paths_state = gr.State([])
        selected_index = gr.State(None)
        counts_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2, min_width=200):
                gr.Markdown("### Dataset")
                gallery = gr.Gallery(
                    label="Images",
                    columns=4,
                    height=420,
                    object_fit="contain",
                    allow_preview=True,
                    show_label=True,
                )
                gallery_cols = gr.Slider(2, 8, value=4, step=1, label="Thumbnail columns")
                showing = gr.Textbox(label="Selection", interactive=False, lines=1)

            with gr.Column(scale=2, min_width=220):
                gr.Markdown("### Image tags")
                current_tags = gr.Textbox(
                    label="Tags for selected image (comma or one per line)",
                    lines=18,
                    placeholder="e.g. 1girl, solo, smile",
                )
                with gr.Row():
                    save_btn = gr.Button("Save to caption file", variant="primary", visible=not headless)
                    reload_btn = gr.Button("Reload from file", visible=not headless)
                with gr.Row():
                    add_box = gr.Textbox(show_label=False, placeholder="New tag to add", scale=3)
                    add_btn = gr.Button("Add", visible=not headless)
                with gr.Row():
                    del_box = gr.Textbox(show_label=False, placeholder="Tag to remove (exact match)", scale=3)
                    del_btn = gr.Button("Remove", visible=not headless)
                dedupe_btn = gr.Button("Remove duplicate tags", visible=not headless)
                with gr.Row():
                    tgt_lang = gr.Dropdown(
                        choices=["zh-CN", "zh-TW", "ja", "en", "ko"],
                        value="zh-CN",
                        label="Translate target",
                    )
                    trans_btn = gr.Button("Translate tags", visible=not headless)
                trans_info = gr.Textbox(label="Translation note", lines=1, interactive=False)

            with gr.Column(scale=2, min_width=220):
                gr.Markdown("### All tags (dataset)")
                filter_g = gr.Textbox(label="Filter tags (contains)", placeholder="substring…")
                global_table = gr.Dataframe(
                    headers=["tag", "count"],
                    datatype=["str", "number"],
                    label="Tag statistics",
                    interactive=False,
                    wrap=True,
                )
                refilter = gr.Button("Apply filter", visible=not headless)

                with gr.Accordion("Batch add tag", open=False):
                    gr.Markdown(
                        "All caption files in folder · same extension as above. "
                        "**Top** = prepend, **Bottom** = append. Skip = leave file unchanged if tag exists."
                    )
                    batch_tag = gr.Textbox(
                        label="Tag to add",
                        placeholder="e.g. masterpiece",
                        lines=1,
                    )
                    batch_pos = gr.Radio(
                        choices=["Top (beginning)", "Bottom (end)"],
                        value="Top (beginning)",
                        label="Adding position",
                    )
                    batch_skip = gr.Checkbox(
                        label="Skip if tag already exists",
                        value=True,
                    )
                    with gr.Row():
                        batch_apply = gr.Button(
                            "Apply to all",
                            variant="primary",
                            visible=not headless,
                            scale=2,
                        )
                        batch_lower = gr.Button(
                            "Lower case",
                            visible=not headless,
                            scale=1,
                        )

                with gr.Accordion("Batch remove tag", open=False):
                    gr.Markdown(
                        "Remove exact-match tag(s) from all caption files in folder. "
                        "Use commas or one tag per line for multiple tags."
                    )
                    batch_remove_tags = gr.Textbox(
                        label="Tag(s) to remove",
                        placeholder="e.g. watermark, blurry",
                        lines=3,
                    )
                    with gr.Row():
                        batch_remove_apply = gr.Button(
                            "Remove from all",
                            variant="stop",
                            visible=not headless,
                            scale=2,
                        )
                        batch_remove_lower = gr.Button(
                            "Lower case",
                            visible=not headless,
                            scale=1,
                        )

        def scan_folder(path: str, ext: str):
            path = (path or "").strip().strip('"')
            if not path or not os.path.isdir(path):
                return [], [], [], "Invalid or empty folder.", [], None
            imgs = _list_images(path)
            pairs = _global_tag_counts(path, ext)
            rows = _rows_from_counts(pairs, "")
            msg = f"Loaded {len(imgs)} images, {len(pairs)} unique tags in *{ext} files."
            log.info(msg)
            return imgs, imgs, rows, msg, pairs, None

        def _select_index(evt: gr.SelectData) -> Optional[int]:
            if evt is None:
                return None
            idx = getattr(evt, "index", None)
            if isinstance(idx, (list, tuple)):
                idx = idx[0] if idx else None
            if idx is None:
                return None
            return int(idx)

        def on_gallery_select(evt: gr.SelectData, paths: List[str], ext: str):
            idx = _select_index(evt)
            if not paths or idx is None or idx < 0 or idx >= len(paths):
                return "", "No selection", None
            ip = paths[idx]
            cap = _caption_path(ip, ext)
            raw = _read_tags_file(cap)
            tags = _join_tags(_parse_tags(raw)) if raw else ""
            base = os.path.basename(ip)
            return tags, f"Selected: {base} ({idx + 1}/{len(paths)})", idx

        def save_current(
            tags: str,
            paths: List[str],
            idx: Optional[int],
            ext: str,
            root: str,
        ):
            root = (root or "").strip().strip('"')
            if not paths:
                return "No dataset loaded.", gr.update(), gr.update()
            if idx is None or idx < 0 or idx >= len(paths):
                return "Select an image in the gallery first.", gr.update(), gr.update()
            if not root or not os.path.isdir(root):
                return "Invalid dataset folder.", gr.update(), gr.update()
            ip = paths[idx]
            cap = _caption_path(ip, ext)
            normalized = _join_tags(_parse_tags(tags))
            _write_tags_file(cap, normalized)
            pairs = _global_tag_counts(root, ext)
            rows = _rows_from_counts(pairs, "")
            msg = f"Saved: {os.path.basename(cap)} — stats refreshed."
            return msg, pairs, rows

        def dedupe_tags(tags: str) -> str:
            seen: List[str] = []
            for t in _parse_tags(tags):
                if t not in seen:
                    seen.append(t)
            return _join_tags(seen)

        def reload_current(paths: List[str], idx: Optional[int], ext: str):
            if not paths or idx is None or idx < 0 or idx >= len(paths):
                return ""
            ip = paths[idx]
            cap = _caption_path(ip, ext)
            raw = _read_tags_file(cap)
            return _join_tags(_parse_tags(raw)) if raw else ""

        def add_tag(tags: str, new: str):
            if not new or not new.strip():
                return tags
            cur = _parse_tags(tags)
            n = new.strip()
            if n not in cur:
                cur.append(n)
            return _join_tags(cur)

        def del_tag(tags: str, rem: str):
            if not rem or not rem.strip():
                return tags
            r = rem.strip()
            cur = [t for t in _parse_tags(tags) if t != r]
            return _join_tags(cur)

        def do_filter(pairs: List[Tuple[str, int]], ft: str):
            return _rows_from_counts(pairs, ft)

        def do_translate(tags: str, target: str):
            out, err = _translate_block(tags, target)
            note = err or ("OK" if _HAS_TRANSLATOR else "deep-translator not installed")
            return out, note

        def set_gallery_columns(cols: float, imgs: List[str]):
            return gr.update(columns=int(cols), value=imgs)

        browse.click(fn=get_folder_path, inputs=folder, outputs=folder)

        refresh.click(
            fn=scan_folder,
            inputs=[folder, caption_ext],
            outputs=[
                gallery,
                image_paths_state,
                global_table,
                status,
                counts_state,
                selected_index,
            ],
        )

        gallery_cols.change(
            fn=set_gallery_columns,
            inputs=[gallery_cols, image_paths_state],
            outputs=gallery,
        )

        gallery.select(
            fn=on_gallery_select,
            inputs=[image_paths_state, caption_ext],
            outputs=[current_tags, showing, selected_index],
        )

        save_btn.click(
            fn=save_current,
            inputs=[current_tags, image_paths_state, selected_index, caption_ext, folder],
            outputs=[status, counts_state, global_table],
        )

        reload_btn.click(
            fn=reload_current,
            inputs=[image_paths_state, selected_index, caption_ext],
            outputs=current_tags,
        )

        add_btn.click(fn=add_tag, inputs=[current_tags, add_box], outputs=current_tags)
        del_btn.click(fn=del_tag, inputs=[current_tags, del_box], outputs=current_tags)
        dedupe_btn.click(fn=dedupe_tags, inputs=current_tags, outputs=current_tags)

        refilter.click(fn=do_filter, inputs=[counts_state, filter_g], outputs=global_table)

        trans_btn.click(fn=do_translate, inputs=[current_tags, tgt_lang], outputs=[current_tags, trans_info])

        batch_lower.click(
            lambda s: (s or "").lower(),
            inputs=batch_tag,
            outputs=batch_tag,
        )
        batch_apply.click(
            fn=_batch_add_tag_apply,
            inputs=[folder, caption_ext, batch_tag, batch_pos, batch_skip],
            outputs=[status, counts_state, global_table],
        )
        batch_remove_lower.click(
            lambda s: (s or "").lower(),
            inputs=batch_remove_tags,
            outputs=batch_remove_tags,
        )
        batch_remove_apply.click(
            fn=_batch_remove_tags_apply,
            inputs=[folder, caption_ext, batch_remove_tags],
            outputs=[status, counts_state, global_table],
        )

        caption_ext.change(
            fn=scan_folder,
            inputs=[folder, caption_ext],
            outputs=[
                gallery,
                image_paths_state,
                global_table,
                status,
                counts_state,
                selected_index,
            ],
        )
