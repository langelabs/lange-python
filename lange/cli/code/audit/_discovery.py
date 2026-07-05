"""Folder and audit-tool discovery helpers for ``lange code audit``."""

from __future__ import annotations

from pathlib import Path

import click

from ._types import AuditTool, PNPM_AUDIT_TOOL, UV_AUDIT_TOOL

PNPM_LOCK_FILES: tuple[str, ...] = ("pnpm-lock", "pnpm-lock.yaml")
UV_LOCK_FILE = "uv.lock"


def list_candidate_folders(root: Path) -> list[Path]:
    """
    List top-level non-hidden directories.

    :param root: Directory that should be scanned.
    :returns: Sorted non-hidden child directories.
    """
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    return sorted(candidates, key=lambda item: item.name.lower())


def resolve_audit_folder(folder_name: str, root: Path) -> Path:
    """
    Resolve an explicitly requested audit folder.

    :param folder_name: Folder argument passed via CLI.
    :param root: Current working directory.
    :returns: Resolved target folder path.
    """
    folder = (root / folder_name).resolve()
    if not folder.exists() or not folder.is_dir():
        raise click.ClickException(
            f"Folder '{folder_name}' was not found or is not a directory."
        )
    return folder


def list_auditable_folders(root: Path) -> list[Path]:
    """
    List top-level folders that contain at least one supported audit lock file.

    :param root: Directory that should be scanned.
    :returns: Sorted list of auditable folders.
    """
    return [
        folder for folder in list_candidate_folders(root) if detect_available_audit_tools(folder)
    ]


def detect_available_audit_tools(folder: Path) -> list[AuditTool]:
    """
    Detect supported audit tools available in the given folder.

    :param folder: Folder that should be inspected.
    :returns: Ordered list of detected audit tools.
    """
    available: list[AuditTool] = []
    if any((folder / lock_file).is_file() for lock_file in PNPM_LOCK_FILES):
        available.append(PNPM_AUDIT_TOOL)
    if (folder / UV_LOCK_FILE).is_file():
        available.append(UV_AUDIT_TOOL)
    return available
