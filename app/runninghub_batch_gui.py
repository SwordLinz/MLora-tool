"""
RunningHub: batch-run a cloud ComfyUI workflow on local images; saves images to a folder.
Shares logic with tools/runninghub_batch_workflow.py and ~/.runninghub_batch_gui.json settings.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator, List, Optional

import gradio as gr

from .class_gui_config import KohyaSSGUIConfig
from .common_gui import get_folder_path, scriptdir
from . import runninghub_batch_workflow as rh

SETTINGS_PATH = Path.home() / ".runninghub_batch_gui.json"
MAX_VISIBLE_LOG_LINES = 30


def _load_json_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.is_file():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_json_settings(payload: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_text(value: Any) -> str:
    return str(value or "").strip().strip('"')


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _settings_payload(
    api_key: str,
    input_dir: str,
    output_dir: str,
    workflow_id: str,
    load_node: str,
    save_node: str,
    seed_node: str,
    max_wait: Any = None,
    poll_interval: Any = None,
    request_timeout: Any = None,
) -> dict[str, Any]:
    saved = _load_json_settings()
    api_key = _clean_text(api_key) or _clean_text(saved.get("api_key"))
    payload = dict(saved)
    payload.update(
        {
            "api_key": api_key,
            "input_dir": _clean_text(input_dir),
            "output_dir": _clean_text(output_dir),
            "workflow_id": _clean_text(workflow_id),
            "load_node": _clean_text(load_node),
            "save_node": _clean_text(save_node),
            "seed_node": _clean_text(seed_node),
        }
    )
    if max_wait is not None:
        payload["max_wait"] = _float_or_default(max_wait, 1800)
    if poll_interval is not None:
        payload["poll_interval"] = _float_or_default(poll_interval, 3)
    if request_timeout is not None:
        payload["request_timeout"] = _float_or_default(request_timeout, 120)
    return payload

def _format_visible_log(lines: List[str]) -> str:
    if len(lines) <= MAX_VISIBLE_LOG_LINES:
        return "\n".join(lines)
    hidden = len(lines) - MAX_VISIBLE_LOG_LINES
    visible = lines[-MAX_VISIBLE_LOG_LINES:]
    return f"... 已隐藏之前 {hidden} 条日志，仅显示最新 {MAX_VISIBLE_LOG_LINES} 条 ...\n" + "\n".join(visible)


def _defaults_from_config_and_json(cfg: KohyaSSGUIConfig) -> dict[str, Any]:
    j = _load_json_settings()
    base_in = cfg.get("utilities.runninghub_input", os.path.join(scriptdir, "data"))
    base_out = cfg.get(
        "utilities.runninghub_output_dir",
        os.path.join(scriptdir, "outputs", "runninghub_out"),
    )
    if j.get("output_dir"):
        out_dir = j["output_dir"]
    elif j.get("output_zip"):
        zp = j["output_zip"]
        out_dir = str(Path(zp).with_suffix("")) if zp.lower().endswith(".zip") else zp
    else:
        out_dir = base_out or ""
    return {
        "api_key": j.get("api_key") or os.environ.get("RUNNINGHUB_API_KEY", ""),
        "input_dir": j.get("input_dir") or base_in or "",
        "output_dir": out_dir,
        "workflow_id": j.get("workflow_id") or rh.DEFAULT_WORKFLOW_ID,
        "load_node": j.get("load_node") or rh.DEFAULT_LOAD_IMAGE_NODE,
        "save_node": j.get("save_node") or rh.DEFAULT_SAVE_NODE,
        "seed_node": j["seed_node"] if "seed_node" in j else rh.DEFAULT_KSAMPLER_NODE,
        "max_wait": j.get("max_wait", 1800),
        "poll_interval": j.get("poll_interval", 3),
        "request_timeout": j.get("request_timeout", 120),
    }


def _run_batch_gradio(
    api_key: str,
    input_dir: str,
    output_dir: str,
    workflow_id: str,
    load_node: str,
    save_node: str,
    seed_node: str,
    max_wait: float,
    poll_interval: float,
    request_timeout: float,
) -> Iterator[str]:
    payload = _settings_payload(
        api_key,
        input_dir,
        output_dir,
        workflow_id,
        load_node,
        save_node,
        seed_node,
        max_wait,
        poll_interval,
        request_timeout,
    )
    try:
        _save_json_settings(payload)
    except OSError as e:
        yield f"保存设置失败: {e}"
        return

    api_key = payload["api_key"]
    input_dir = payload["input_dir"]
    output_dir = payload["output_dir"]
    workflow_id = payload["workflow_id"] or rh.DEFAULT_WORKFLOW_ID
    load_node = payload["load_node"] or rh.DEFAULT_LOAD_IMAGE_NODE
    save_node = payload["save_node"]
    seed_node = payload["seed_node"]
    max_wait = payload["max_wait"]
    poll_interval = payload["poll_interval"]
    request_timeout = payload["request_timeout"]
    if not api_key:
        yield "错误：请填写 API Key（或设置环境变量 RUNNINGHUB_API_KEY）。"
        return
    if not input_dir or not os.path.isdir(input_dir):
        yield "错误：请选择有效的输入图片文件夹。"
        return
    if not output_dir:
        yield "错误：请指定输出图片文件夹。"
        return

    out_path = Path(output_dir)

    log_q: queue.Queue[str] = queue.Queue()
    progress_q: queue.Queue[str] = queue.Queue()
    done = threading.Event()
    result: list[Any] = [None]

    def log_fn(s: str) -> None:
        log_q.put(s)

    def progress_fn(s: str) -> None:
        progress_q.put(s)

    def worker() -> None:
        try:
            p = rh.run_batch(
                api_key,
                Path(input_dir),
                out_path,
                workflow_id=workflow_id,
                load_image_node=load_node,
                save_node=save_node,
                random_seed_node=seed_node,
                poll_interval=float(poll_interval),
                max_wait=float(max_wait),
                request_timeout=float(request_timeout),
                log=log_fn,
                progress=progress_fn,
            )
            result[0] = p
        except Exception as e:
            result[0] = e
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()
    lines: List[str] = []
    current_progress = ""
    last_heartbeat = 0.0
    while not done.is_set() or not log_q.empty() or not progress_q.empty():
        changed = False
        while not log_q.empty():
            try:
                lines.append(log_q.get_nowait())
                changed = True
            except queue.Empty:
                break
        while not progress_q.empty():
            try:
                current_progress = progress_q.get_nowait()
                changed = True
            except queue.Empty:
                break
        now = time.monotonic()
        if changed or now - last_heartbeat >= 5.0:
            last_heartbeat = now
            text = _format_visible_log(lines)
            if current_progress:
                text = f"{text}\n\n当前轮询状态（仅显示最新一条）:\n{current_progress}"
            yield text
        if not done.is_set():
            time.sleep(0.25)

    while not log_q.empty():
        lines.append(log_q.get_nowait())
    while not progress_q.empty():
        current_progress = progress_q.get_nowait()
    err = result[0]
    if isinstance(err, Exception):
        lines.append(f"错误: {err}")
    elif err is not None:
        lines.append(f"完成: {err}")
    yield _format_visible_log(lines)


def _save_clicked(
    api_key: str,
    input_dir: str,
    output_dir: str,
    workflow_id: str,
    load_node: str,
    save_node: str,
    seed_node: str,
    max_wait: float,
    poll_interval: float,
    request_timeout: float,
) -> str:
    try:
        _save_json_settings(
            _settings_payload(
                api_key,
                input_dir,
                output_dir,
                workflow_id,
                load_node,
                save_node,
                seed_node,
                max_wait,
                poll_interval,
                request_timeout,
            )
        )
        return f"已保存到 {SETTINGS_PATH}"
    except OSError as e:
        return f"保存失败: {e}"

def gradio_runninghub_batch_tab(
    headless: bool = False,
    config: Optional[KohyaSSGUIConfig] = None,
) -> None:
    cfg = config or KohyaSSGUIConfig()
    d = _defaults_from_config_and_json(cfg)

    with gr.Tab("RunningHub 批量"):
        gr.Markdown(
            "对文件夹内每张图调用 [RunningHub](https://www.runninghub.cn) 云端 ComfyUI 工作流，"
            "生成图片、latent 等输出文件直接保存到下方输出文件夹。需开通 API 并在网页上至少成功运行过该工作流一次。"
            " [API 说明](https://www.runninghub.cn/runninghub-api-doc-cn/)"
        )

        api_key = gr.Textbox(
            label="API Key",
            value=d["api_key"],
            type="password",
            placeholder="RunningHub API 控制台获取",
        )
        input_folder = gr.Textbox(label="输入图片文件夹", value=d["input_dir"], placeholder="含 jpg/png/webp 的目录")
        output_folder = gr.Textbox(
            label="输出文件夹",
            value=d["output_dir"],
            placeholder="结果保存到此目录（可新建；不存在会自动创建）",
        )

        with gr.Row():
            in_browse = gr.Button("📂 输入目录", elem_classes=["tool"], visible=not headless)
            out_browse = gr.Button("📂 输出目录", elem_classes=["tool"], visible=not headless)

        with gr.Accordion("工作流节点（与网页节点右上角 ID 一致）", open=False):
            workflow_id = gr.Textbox(label="Workflow ID", value=d["workflow_id"])
            load_node = gr.Textbox(label="Load Image 节点 ID", value=d["load_node"])
            save_node = gr.Textbox(label="保存/输出节点 ID（留空=保留全部支持的文件输出）", value=d["save_node"])
            seed_node = gr.Textbox(
                label="KSampler 随机种子节点 ID（留空=不随机）",
                value=d["seed_node"],
            )

        with gr.Accordion("高级", open=False):
            max_wait = gr.Number(value=d["max_wait"], label="单张最长等待（秒）", precision=0)
            poll_interval = gr.Number(value=d["poll_interval"], label="轮询间隔（秒）", precision=1)
            request_timeout = gr.Number(value=d["request_timeout"], label="单次 HTTP 超时（秒）", precision=0)

        with gr.Row():
            run_btn = gr.Button("开始批量处理", variant="primary", visible=not headless)
            save_btn = gr.Button("保存设置", visible=not headless)

        log = gr.Textbox(label="日志", lines=30, max_lines=30, interactive=False)
        save_status = gr.Textbox(label="保存提示", lines=1, interactive=False, visible=not headless)

        in_browse.click(fn=get_folder_path, inputs=input_folder, outputs=input_folder)
        out_browse.click(fn=get_folder_path, inputs=output_folder, outputs=output_folder)

        run_btn.click(
            fn=_run_batch_gradio,
            inputs=[
                api_key,
                input_folder,
                output_folder,
                workflow_id,
                load_node,
                save_node,
                seed_node,
                max_wait,
                poll_interval,
                request_timeout,
            ],
            outputs=log,
        )

        save_btn.click(
            fn=_save_clicked,
            inputs=[api_key, input_folder, output_folder, workflow_id, load_node, save_node, seed_node, max_wait, poll_interval, request_timeout],
            outputs=save_status,
        )
