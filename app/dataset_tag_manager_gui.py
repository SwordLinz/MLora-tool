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
from .gradio_paths import allow_path

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
        return text, "请先安装：pip install deep-translator"
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
        return "文件夹无效。", [], []
    if not tag:
        return "请输入要添加的标签。", [], []
    pos = (position_label or "")
    top = "开头" in pos or "最前" in pos or "前" in pos[:2]
    paths = _list_caption_files(root, ext)
    if not paths:
        return f"文件夹中没有 *{ext} 标签文件。", [], []
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
        f"批量添加：已更新 {changed} 个文件，跳过 {skipped} 个（标签已存在）。"
        f"插入位置：{'开头' if top else '结尾'}。"
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
        return "文件夹无效。", [], []
    if not tags_to_remove:
        return "请输入要移除的一个或多个标签。", [], []
    paths = _list_caption_files(root, ext)
    if not paths:
        return f"文件夹中没有 *{ext} 标签文件。", [], []

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
        f"批量移除：从 {changed} 个文件中移除了 {removed} 处标签。"
        f"标签：{', '.join(sorted(tags_to_remove, key=str.lower))}。"
    )
    log.info(msg)
    return msg, pairs, rows


def gradio_dataset_tag_manager_tab(
    headless: bool = False,
    config: Optional[KohyaSSGUIConfig] = None,
):
    cfg = config or KohyaSSGUIConfig()
    default_dir = cfg.get("utilities.dataset_tag_manager_dir", os.path.join(scriptdir, "data"))

    with gr.Tab("标签管理"):
        gr.Markdown(
            "浏览带有同名标签文件的图片文件夹（`.txt` / `.caption`）。"
            "左栏：缩略图；中栏：所选图片的标签；右栏：全部标签文件的标签频率统计。"
            "标签以逗号分隔文本保存（kohya / danbooru 风格）。"
        )

        folder = gr.Textbox(
            label="数据集文件夹",
            value=default_dir or "",
            placeholder="包含图片和标签文件的文件夹",
        )
        caption_ext = gr.Dropdown(
            choices=[".txt", ".caption"],
            value=".txt",
            label="标签文件扩展名",
        )
        with gr.Row():
            browse = gr.Button("📂", elem_classes=["tool"], visible=not headless)
            refresh = gr.Button("刷新", visible=not headless)
        status = gr.Textbox(label="状态", lines=1, interactive=False)

        image_paths_state = gr.State([])
        selected_index = gr.State(None)
        counts_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=2, min_width=200):
                gr.Markdown("### 数据集")
                gallery = gr.Gallery(
                    label="图片",
                    columns=4,
                    height=420,
                    object_fit="contain",
                    allow_preview=True,
                    show_label=True,
                )
                gallery_cols = gr.Slider(2, 8, value=4, step=1, label="缩略图列数")
                showing = gr.Textbox(label="当前选中", interactive=False, lines=1)

            with gr.Column(scale=2, min_width=220):
                gr.Markdown("### 图片标签")
                current_tags = gr.Textbox(
                    label="所选图片的标签（逗号分隔或每行一个）",
                    lines=18,
                    placeholder="例如：1girl, solo, smile",
                )
                with gr.Row():
                    save_btn = gr.Button("保存到标签文件", variant="primary", visible=not headless)
                    reload_btn = gr.Button("从文件重新加载", visible=not headless)
                with gr.Row():
                    add_box = gr.Textbox(show_label=False, placeholder="要添加的新标签", scale=3)
                    add_btn = gr.Button("添加", visible=not headless)
                with gr.Row():
                    del_box = gr.Textbox(show_label=False, placeholder="要移除的标签（精确匹配）", scale=3)
                    del_btn = gr.Button("移除", visible=not headless)
                dedupe_btn = gr.Button("去除重复标签", visible=not headless)
                with gr.Row():
                    tgt_lang = gr.Dropdown(
                        choices=["zh-CN", "zh-TW", "ja", "en", "ko"],
                        value="zh-CN",
                        label="翻译目标语言",
                    )
                    trans_btn = gr.Button("翻译标签", visible=not headless)
                trans_info = gr.Textbox(label="翻译提示", lines=1, interactive=False)

            with gr.Column(scale=2, min_width=220):
                gr.Markdown("### 全部标签（数据集）")
                filter_g = gr.Textbox(label="过滤标签（包含）", placeholder="输入关键字…")
                global_table = gr.Dataframe(
                    headers=["标签", "次数"],
                    datatype=["str", "number"],
                    label="标签统计",
                    interactive=False,
                    wrap=True,
                )
                refilter = gr.Button("应用过滤", visible=not headless)

                with gr.Accordion("批量添加标签", open=False):
                    gr.Markdown(
                        "作用于文件夹内全部标签文件 · 扩展名与上方一致。"
                        "**开头** = 插入最前，**结尾** = 追加最后。跳过 = 文件中已有该标签时不修改。"
                    )
                    batch_tag = gr.Textbox(
                        label="要添加的标签",
                        placeholder="例如：masterpiece",
                        lines=1,
                    )
                    batch_pos = gr.Radio(
                        choices=["开头（最前）", "结尾（最后）"],
                        value="开头（最前）",
                        label="插入位置",
                    )
                    batch_skip = gr.Checkbox(
                        label="标签已存在时跳过",
                        value=True,
                    )
                    with gr.Row():
                        batch_apply = gr.Button(
                            "应用到全部",
                            variant="primary",
                            visible=not headless,
                            scale=2,
                        )
                        batch_lower = gr.Button(
                            "转小写",
                            visible=not headless,
                            scale=1,
                        )

                with gr.Accordion("批量移除标签", open=False):
                    gr.Markdown(
                        "从文件夹内全部标签文件中移除精确匹配的标签。"
                        "多个标签用逗号分隔或每行一个。"
                    )
                    batch_remove_tags = gr.Textbox(
                        label="要移除的标签",
                        placeholder="例如：watermark, blurry",
                        lines=3,
                    )
                    with gr.Row():
                        batch_remove_apply = gr.Button(
                            "从全部文件移除",
                            variant="stop",
                            visible=not headless,
                            scale=2,
                        )
                        batch_remove_lower = gr.Button(
                            "转小写",
                            visible=not headless,
                            scale=1,
                        )

        def scan_folder(path: str, ext: str):
            path = (path or "").strip().strip('"')
            if not path or not os.path.isdir(path):
                return [], [], [], "文件夹无效或为空。", [], None
            imgs = _list_images(path)
            allow_path(path)
            pairs = _global_tag_counts(path, ext)
            rows = _rows_from_counts(pairs, "")
            msg = f"已加载 {len(imgs)} 张图片，*{ext} 文件中共 {len(pairs)} 个不同标签。"
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
                return "", "未选中任何图片", None
            ip = paths[idx]
            cap = _caption_path(ip, ext)
            raw = _read_tags_file(cap)
            tags = _join_tags(_parse_tags(raw)) if raw else ""
            base = os.path.basename(ip)
            return tags, f"已选中：{base}（{idx + 1}/{len(paths)}）", idx

        def save_current(
            tags: str,
            paths: List[str],
            idx: Optional[int],
            ext: str,
            root: str,
        ):
            root = (root or "").strip().strip('"')
            if not paths:
                return "尚未加载数据集。", gr.update(), gr.update()
            if idx is None or idx < 0 or idx >= len(paths):
                return "请先在图库中选择一张图片。", gr.update(), gr.update()
            if not root or not os.path.isdir(root):
                return "数据集文件夹无效。", gr.update(), gr.update()
            ip = paths[idx]
            cap = _caption_path(ip, ext)
            normalized = _join_tags(_parse_tags(tags))
            _write_tags_file(cap, normalized)
            pairs = _global_tag_counts(root, ext)
            rows = _rows_from_counts(pairs, "")
            msg = f"已保存：{os.path.basename(cap)} — 统计已刷新。"
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
            note = err or ("完成" if _HAS_TRANSLATOR else "未安装 deep-translator")
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
