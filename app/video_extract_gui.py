"""
Batch extract frames from video files to a folder (optional subfolder per video).
Uses OpenCV (cv2).
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional, Tuple

import gradio as gr

from .class_gui_config import KohyaSSGUIConfig
from .common_gui import get_folder_path, scriptdir
from .custom_logging import setup_logging

log = setup_logging()

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv", ".flv", ".m4v", ".mpg", ".mpeg")

_WIN_BAD = set('<>:"/\\|?*\n\r\t')


def _safe_folder_segment(name: str, max_len: int = 48) -> str:
    """Strip characters invalid in Windows folder names; trim length."""
    s = name.replace("\n", " ").strip()[:max_len]
    s = "".join("_" if c in _WIN_BAD else c for c in s)
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s or "video"


def _subfolder_name_for_video(base_name: str) -> str:
    """
    Short unique folder name — full video filenames are often >200 chars and break Windows MAX_PATH
    when combined with output path + frame file name.
    """
    h = hashlib.sha256(base_name.encode("utf-8")).hexdigest()[:12]
    prefix = _safe_folder_segment(base_name, 40)
    return f"{prefix}_{h}"


def _win_long_path(path: str) -> str:
    """Enable extended-length paths on Windows (beyond 260 chars) for makedirs/open."""
    if os.name != "nt":
        return path
    abs_path = os.path.abspath(path).replace("/", "\\")
    if abs_path.startswith("\\\\?\\"):
        return abs_path
    if abs_path.startswith("\\\\"):
        # UNC \\server\share — must not treat \\?\ as UNC (check order above)
        return "\\\\?\\UNC\\" + abs_path[2:]
    return "\\\\?\\" + abs_path


def _save_frame_bgr(path: str, frame_bgr, ext: str) -> bool:
    """
    Write BGR image to path. Uses imencode + binary write so Unicode / long paths work on Windows
    (cv2.imwrite often fails silently for non-ASCII paths).
    """
    try:
        import cv2
    except ImportError:
        return False
    ext = ext.lower()
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        )
    elif ext == ".png":
        ok, buf = cv2.imencode(".png", frame_bgr)
    else:
        ok, buf = cv2.imencode(ext if ext.startswith(".") else f".{ext}", frame_bgr)
    if not ok or buf is None:
        return False
    try:
        wpath = _win_long_path(path)
        parent = os.path.dirname(wpath)
        if parent and parent != wpath:
            os.makedirs(parent, exist_ok=True)
        with open(wpath, "wb") as f:
            f.write(buf.tobytes())
        return True
    except OSError as e:
        log.warning("Could not write frame %s: %s", path, e)
        return False


def _list_videos(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(folder), key=str.lower):
        low = name.lower()
        if any(low.endswith(e) for e in VIDEO_EXTS):
            out.append(os.path.join(folder, name))
    return out


def _extract_one_video(
    video_path: str,
    out_base: str,
    mode: str,
    value: float,
    img_format: str,
    subfolder: bool,
) -> Tuple[int, str]:
    """
    Returns (saved_count, error_message_or_empty).
    """
    try:
        import cv2
    except ImportError:
        return 0, "OpenCV (cv2) is not available."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, f"Cannot open video: {video_path}"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    if subfolder:
        sub = _subfolder_name_for_video(base_name)
        out_dir = os.path.join(out_base, sub)
    else:
        out_dir = out_base
    os.makedirs(_win_long_path(out_dir), exist_ok=True)

    ext = ".jpg" if img_format.lower() in ("jpg", "jpeg") else ".png"
    mode_key = (mode or "").strip()
    frame_idx = 0
    saved = 0
    write_attempts = 0
    out_seq = 0
    last_save_time = -1.0

    def _write(frame_bgr) -> None:
        nonlocal saved, write_attempts, out_seq
        write_attempts += 1
        seq = out_seq
        out_seq += 1
        # Subfolder already names the video — use short names to avoid Windows MAX_PATH / long filenames.
        if subfolder:
            fname = f"frame_{seq:06d}{ext}"
        else:
            safe = base_name[:100] if len(base_name) > 100 else base_name
            fname = f"{safe}_{seq:06d}{ext}"
        fpath = os.path.join(out_dir, fname)
        if _save_frame_bgr(fpath, frame_bgr, ext):
            saved += 1
        else:
            log.warning("Frame write failed (skipped): %s", fpath)

    if mode_key == "Every N frames":
        n = max(1, int(round(float(value))))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % n == 0:
                _write(frame)
            frame_idx += 1
    elif mode_key == "Every N seconds":
        sec = max(0.01, float(value))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_idx / fps
            if last_save_time < 0 or t - last_save_time >= sec - 1e-9:
                _write(frame)
                last_save_time = t
            frame_idx += 1
    else:
        # Images per second of video (target sample rate)
        rate = max(0.01, float(value))
        interval = max(1, int(round(fps / rate)))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                _write(frame)
            frame_idx += 1

    cap.release()
    if write_attempts > 0 and saved == 0:
        return 0, (
            "All frame writes failed (path too long or permission). "
            "Subfolders now use a short hash name; output also uses Windows long-path mode."
        )
    return saved, ""


def _run_batch(
    input_dir: str,
    output_dir: str,
    mode: str,
    value: float,
    img_format: str,
    subfolder: bool,
) -> str:
    input_dir = (input_dir or "").strip().strip('"')
    output_dir = (output_dir or "").strip().strip('"')
    if not input_dir or not os.path.isdir(input_dir):
        return "Invalid input folder."
    if not output_dir:
        return "Set output folder."
    os.makedirs(_win_long_path(output_dir), exist_ok=True)

    videos = _list_videos(input_dir)
    if not videos:
        return "No video files found in input folder."

    total_saved = 0
    errors: List[str] = []
    for vp in videos:
        n, err = _extract_one_video(vp, output_dir, mode, value, img_format, subfolder)
        total_saved += n
        if err:
            errors.append(err)
        log.info("Extracted %s frames from %s", n, os.path.basename(vp))

    msg = f"Done. Saved {total_saved} image(s) from {len(videos)} video(s) → {output_dir}"
    if errors:
        msg += " | Errors: " + "; ".join(errors[:3])
    return msg


def gradio_video_extract_tab(
    headless: bool = False,
    config: Optional[KohyaSSGUIConfig] = None,
):
    cfg = config or KohyaSSGUIConfig()
    default_in = cfg.get("utilities.video_extract_input", os.path.join(scriptdir, "data"))
    default_out = cfg.get("utilities.video_extract_output", os.path.join(scriptdir, "outputs", "video_frames"))

    with gr.Tab("Video to Images"):
        gr.Markdown(
            "Extract image sequences from all videos in a folder. "
            "Use **Every N frames** for fixed step, **Every N seconds** for time-based sampling, "
            "or **Images per second** to target a frame rate (e.g. 1 ≈ one image per second of video). "
            "Optional: **subfolder per video** uses a **short name + hash** (not the full filename) so Windows path limits are not exceeded."
        )

        input_folder = gr.Textbox(label="Input folder (videos)", value=default_in or "", placeholder="Folder containing video files")
        output_folder = gr.Textbox(
            label="Output folder",
            value=default_out or "",
            placeholder="Extracted images saved here",
        )
        with gr.Row():
            in_browse = gr.Button("📂", elem_classes=["tool"], visible=not headless)
            out_browse = gr.Button("📂 out", elem_classes=["tool"], visible=not headless)

        subfolder = gr.Checkbox(
            label="Create subfolder per video (short name + hash; avoids Windows path limit)",
            value=True,
        )
        mode = gr.Radio(
            choices=[
                "Every N frames",
                "Every N seconds",
                "Images per second of video",
            ],
            value="Images per second of video",
            label="Extraction mode",
        )
        value_num = gr.Number(
            value=1.0,
            precision=2,
            label="Value (N frames / N seconds / images per second, depending on mode)",
        )
        img_format = gr.Dropdown(choices=["png", "jpg"], value="png", label="Output image format")

        run_btn = gr.Button("Extract frames", variant="primary", visible=not headless)
        status = gr.Textbox(label="Status", interactive=False, lines=3)

        in_browse.click(fn=get_folder_path, inputs=input_folder, outputs=input_folder)
        out_browse.click(fn=get_folder_path, inputs=output_folder, outputs=output_folder)

        run_btn.click(
            fn=_run_batch,
            inputs=[input_folder, output_folder, mode, value_num, img_format, subfolder],
            outputs=status,
        )
