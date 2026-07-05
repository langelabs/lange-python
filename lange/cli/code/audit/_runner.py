"""Command execution helpers for ``lange code audit``."""

from __future__ import annotations

import subprocess
from pathlib import Path

from ._types import AuditTool, PNPM_AUDIT_TOOL, UV_AUDIT_TOOL


def build_audit_command(audit_tool: AuditTool) -> list[str]:
    """
    Build the subprocess command for one supported audit tool.

    :param audit_tool: Tool that should be executed.
    :returns: Command arguments for ``subprocess.run``.
    """
    if audit_tool == PNPM_AUDIT_TOOL:
        return ["pnpm", "audit"]
    return ["uv", "audit"]


def run_audit_command(folder: Path, audit_tool: AuditTool) -> None:
    """
    Execute one audit command in the given folder.

    :param folder: Working directory for the audit command.
    :param audit_tool: Tool that should be executed.
    :returns: ``None``.
    """
    subprocess.run(build_audit_command(audit_tool), check=True, cwd=folder)
