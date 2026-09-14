"""
Runtime allow-listing of user-chosen folders for Gradio.

Gradio only serves/caches files that live in the launch working directory, the
system temp directory, or a path explicitly listed in ``Blocks.allowed_paths``.
This tool lets the user pick any folder at runtime, so folders have to be
registered while the app is already running. ``_check_allowed`` reads
``LocalContext.blocks.get(None).allowed_paths`` on every request, so appending to
that list takes effect immediately for both cache moves and ``/file=`` serving.
"""
from __future__ import annotations

import os
from typing import Optional


def allow_path(path: Optional[str]) -> None:
    """Register ``path`` (a file or folder) so Gradio may serve it.

    No-op outside a running Gradio callback and for empty/missing paths. Passing a
    file registers its parent directory, which is what Gradio checks against.
    """
    if not path:
        return
    try:
        from gradio.context import LocalContext
    except ImportError:  # pragma: no cover - gradio is a hard dependency
        return
    blocks = LocalContext.blocks.get(None)
    if blocks is None:
        return
    try:
        abs_path = os.path.abspath(str(path).strip().strip('"'))
    except (OSError, ValueError):
        return
    if not abs_path:
        return
    folder = abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path)
    if folder and folder not in blocks.allowed_paths:
        blocks.allowed_paths.append(folder)
