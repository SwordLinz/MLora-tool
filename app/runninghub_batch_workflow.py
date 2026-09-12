"""
Batch-run a RunningHub ComfyUI workflow on local images; saves output files in a folder.

Requires: pip install requests (usually already present with gradio).

API reference: https://www.runninghub.cn/runninghub-api-doc-cn/
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

try:
    import requests
except ImportError:
    requests = None  # type: ignore[misc, assignment]

BASE = "https://www.runninghub.cn"
UPLOAD_URL = f"{BASE}/openapi/v2/media/upload/binary"
CREATE_URL = f"{BASE}/task/openapi/create"
OUTPUTS_URL = f"{BASE}/task/openapi/outputs"

DEFAULT_WORKFLOW_ID = "2039022330722131970"
DEFAULT_LOAD_IMAGE_NODE = "1153"
DEFAULT_KSAMPLER_NODE = "185"
DEFAULT_SAVE_NODE = "234"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_FILE_TYPES = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "latent"}


def _require_requests():
    if requests is None:
        raise RuntimeError("需要安装 requests 库: pip install requests")
    return requests


def _headers_json(api_key: str) -> dict[str, str]:
    return {
        "Host": "www.runninghub.cn",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def upload_image(api_key: str, path: Path, timeout: float) -> str:
    """Upload a local image; return `fileName` for Load Image node."""
    req = _require_requests()
    with path.open("rb") as f:
        r = req.post(
            UPLOAD_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (path.name, f, "application/octet-stream")},
            timeout=timeout,
        )
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"Upload failed: {body.get('message') or body.get('msg') or body}")
    data = body.get("data") or {}
    name = data.get("fileName")
    if not name:
        raise RuntimeError(f"Upload response missing fileName: {body}")
    return str(name)


def create_task(
    api_key: str,
    workflow_id: str,
    node_info_list: list[dict[str, Any]],
    timeout: float,
) -> str:
    payload: dict[str, Any] = {
        "apiKey": api_key,
        "workflowId": workflow_id,
        "nodeInfoList": node_info_list,
    }
    req = _require_requests()
    r = req.post(
        CREATE_URL,
        headers=_headers_json(api_key),
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    if body.get("code") != 0:
        raise RuntimeError(f"create task failed: {body.get('msg') or body}")
    data = body.get("data") or {}
    tid = data.get("taskId")
    if tid is None:
        raise RuntimeError(f"create task missing taskId: {body}")
    return str(tid)


def fetch_outputs(api_key: str, task_id: str, timeout: float) -> tuple[int, str, Any]:
    """
    Returns (code, msg, data).
    code 0: success, data is list of output dicts.
    code 804: still running, 813: queued — caller should retry.
    """
    req = _require_requests()
    r = req.post(
        OUTPUTS_URL,
        headers=_headers_json(api_key),
        json={"apiKey": api_key, "taskId": task_id},
        timeout=timeout,
    )
    r.raise_for_status()
    body = r.json()
    return int(body.get("code", -1)), str(body.get("msg") or ""), body.get("data")


def wait_for_outputs(
    api_key: str,
    task_id: str,
    poll_interval: float,
    request_timeout: float,
    max_wait: float | None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    deadline = None if max_wait is None else time.monotonic() + max_wait
    started_at = time.monotonic()
    last_progress_log = 0.0
    while True:
        code, msg, data = fetch_outputs(api_key, task_id, request_timeout)
        if code == 0:
            if not isinstance(data, list):
                raise RuntimeError(f"outputs 返回了意外的数据格式：{data!r}")
            if log:
                elapsed = int(time.monotonic() - started_at)
                log(f"  输出已就绪，耗时 {elapsed} 秒，共 {len(data)} 个文件")
            if progress:
                progress("")
            return data
        if code in (804, 813):
            elapsed = time.monotonic() - started_at
            if progress and elapsed - last_progress_log >= max(10.0, poll_interval):
                last_progress_log = elapsed
                progress(
                    f"  仍在等待 taskId={task_id}，code={code}，"
                    f"状态={msg or '运行中'}，已等待 {int(elapsed)} 秒"
                )
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"任务 {task_id} 超时仍未完成，最后状态：{msg}")
            time.sleep(poll_interval)
            continue
        if code == 805:
            raise RuntimeError(f"任务失败：{msg} data={data!r}")
        raise RuntimeError(f"outputs API 错误 code={code} msg={msg} data={data!r}")


def download_file(url: str, dest: Path, timeout: float) -> None:
    req = _require_requests()
    r = req.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


def output_extension(item: dict[str, Any]) -> str | None:
    file_type = str(item.get("fileType") or "").strip().lower().lstrip(".")
    if "/" in file_type:
        file_type = file_type.rsplit("/", 1)[-1]
    if file_type == "jpeg":
        return "jpg"
    if file_type in OUTPUT_FILE_TYPES:
        return file_type

    for key in ("fileName", "fileUrl"):
        value = item.get(key)
        if not value:
            continue
        path = urlparse(str(value)).path
        suffix = Path(path).suffix.lower().lstrip(".")
        if suffix == "jpeg":
            return "jpg"
        if suffix in OUTPUT_FILE_TYPES:
            return suffix
    return None


def collect_images(folder: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


def run_batch(
    api_key: str,
    input_dir: Path,
    output_dir: Path,
    *,
    workflow_id: str = DEFAULT_WORKFLOW_ID,
    load_image_node: str = DEFAULT_LOAD_IMAGE_NODE,
    save_node: str = DEFAULT_SAVE_NODE,
    random_seed_node: str = DEFAULT_KSAMPLER_NODE,
    poll_interval: float = 3.0,
    max_wait: float = 1800.0,
    request_timeout: float = 120.0,
    log: Callable[[str], None] | None = None,
    progress: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """
    Process each image in ``input_dir`` and save generated images under ``output_dir``.
    Returns the resolved output directory path.
    """
    _log = log or (lambda s: print(s, flush=True))
    _progress = progress or (lambda s: None)
    _cancel = should_cancel or (lambda: False)

    if not api_key.strip():
        raise ValueError("API Key 为空")
    if not input_dir.is_dir():
        raise ValueError(f"不是有效文件夹: {input_dir}")

    images = collect_images(input_dir)
    if not images:
        raise ValueError(f"文件夹内没有支持的图片: {', '.join(sorted(IMAGE_SUFFIXES))}")

    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_node_filter = (save_node or "").strip()
    seed_node = (random_seed_node or "").strip()

    for idx, img_path in enumerate(images):
        if _cancel():
            raise InterruptedError("已取消")
        prefix = f"{idx:04d}_{img_path.stem}"
        _log(f"[{idx + 1}/{len(images)}] 上传 {img_path.name} ...")
        file_name = upload_image(api_key, img_path, request_timeout)

        node_info: list[dict[str, Any]] = [
            {
                "nodeId": str(load_image_node),
                "fieldName": "image",
                "fieldValue": file_name,
            }
        ]
        if seed_node:
            node_info.append(
                {
                    "nodeId": seed_node,
                    "fieldName": "seed",
                    "fieldValue": random.randint(0, 2**48 - 1),
                }
            )

        task_id = create_task(api_key, str(workflow_id), node_info, request_timeout)
        _log(f"  taskId={task_id}，等待完成 ...")
        outputs = wait_for_outputs(
            api_key,
            task_id,
            poll_interval,
            request_timeout,
            max_wait,
            log=_log,
            progress=_progress,
        )

        downloadable: list[dict[str, Any]] = []
        picked: list[dict[str, Any]] = []
        output_node_ids: set[str] = set()
        for item in outputs:
            if not isinstance(item, dict):
                continue
            nid = str(item.get("nodeId", ""))
            if nid:
                output_node_ids.add(nid)
            url = item.get("fileUrl")
            ext = output_extension(item)
            if not url or not ext:
                continue
            item["_output_ext"] = ext
            downloadable.append(item)
            if not save_node_filter or nid == save_node_filter:
                picked.append(item)

        if not picked and save_node_filter and downloadable:
            nodes = ", ".join(sorted(output_node_ids)) or "unknown"
            _log(
                f"  指定保存节点 {save_node_filter} 没有返回可下载文件；"
                f"改为保存本次任务的全部可下载输出。返回节点: {nodes}"
            )
            picked = downloadable

        if not picked:
            nodes = ", ".join(sorted(output_node_ids)) or "none"
            raise RuntimeError(
                f"任务 {task_id} 没有可用的文件输出。"
                f"返回节点: {nodes}。可尝试清空「保存节点」或检查节点 ID。原始: {outputs!r}"
            )
        for j, item in enumerate(picked):
            url = item["fileUrl"]
            ext = item.get("_output_ext") or output_extension(item) or "bin"
            inner_name = f"{prefix}_out{j + 1}.{ext}" if len(picked) > 1 else f"{prefix}.{ext}"
            dest = out_dir / inner_name
            download_file(url, dest, request_timeout)
            _log(f"  已保存 {inner_name}")

    _log(f"完成: {out_dir}（共 {len(images)} 张原图）")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run a RunningHub workflow on each image in a folder; save outputs in a directory."
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("RUNNINGHUB_API_KEY", ""),
        help="RunningHub API Key (or set RUNNINGHUB_API_KEY)",
    )
    p.add_argument(
        "--workflow-id",
        default=DEFAULT_WORKFLOW_ID,
        help=f"Workflow ID from the URL (default: {DEFAULT_WORKFLOW_ID})",
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Folder containing input images")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder to write generated output files (created if missing)",
    )
    p.add_argument(
        "--load-image-node",
        default=DEFAULT_LOAD_IMAGE_NODE,
        help=f"Load Image node id for image input (default {DEFAULT_LOAD_IMAGE_NODE})",
    )
    p.add_argument(
        "--save-node",
        default=DEFAULT_SAVE_NODE,
        help=f"Only include outputs from this save/output node id (default {DEFAULT_SAVE_NODE}); "
        "use empty string to keep all supported file outputs",
    )
    p.add_argument(
        "--random-seed-node",
        default=DEFAULT_KSAMPLER_NODE,
        help=f"If set, randomize KSampler seed on this node id each run (default {DEFAULT_KSAMPLER_NODE}); "
        "use empty string to disable",
    )
    p.add_argument("--poll-interval", type=float, default=3.0, help="Seconds between status checks")
    p.add_argument(
        "--max-wait",
        type=float,
        default=1800.0,
        help="Max seconds to wait per image (default 1800)",
    )
    p.add_argument("--request-timeout", type=float, default=120.0, help="HTTP timeout per request")
    args = p.parse_args()

    if requests is None:
        print("Error: install requests first: pip install requests", file=sys.stderr)
        raise SystemExit(1)

    if not args.api_key.strip():
        print("Error: pass --api-key or set RUNNINGHUB_API_KEY", file=sys.stderr)
        raise SystemExit(2)

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        raise SystemExit(2)

    images = collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir} ({', '.join(sorted(IMAGE_SUFFIXES))})", file=sys.stderr)
        raise SystemExit(2)

    out_dir: Path = args.output_dir
    try:
        run_batch(
            args.api_key,
            input_dir,
            out_dir,
            workflow_id=str(args.workflow_id),
            load_image_node=str(args.load_image_node),
            save_node=str(args.save_node),
            random_seed_node=str(args.random_seed_node),
            poll_interval=args.poll_interval,
            max_wait=args.max_wait,
            request_timeout=args.request_timeout,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
