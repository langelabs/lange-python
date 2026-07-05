"""CLI command orchestration for ``lange code audit``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from ._discovery import (
    detect_available_audit_tools,
    list_auditable_folders,
    resolve_audit_folder,
)
from ._runner import run_audit_command


@click.command("audit")
@click.argument("folder_name", required=False)
def code_audit(folder_name: str | None) -> None:
    """
    Run dependency audits for one folder or all auditable top-level folders.

    :param folder_name: Optional folder name to audit.
    :returns: ``None``.
    """
    if folder_name:
        target_folder = resolve_audit_folder(folder_name=folder_name, root=Path.cwd())
        run_audits_for_folder(folder=target_folder)
        return

    target_folders = list_auditable_folders(Path.cwd())
    if not target_folders:
        raise click.ClickException(
            "No auditable services were found in the current directory."
        )

    for target_folder in target_folders:
        run_audits_for_folder(folder=target_folder)


def run_audits_for_folder(folder: Path) -> None:
    """
    Detect and execute all supported audit commands for one folder.

    :param folder: Folder that should be audited.
    :returns: ``None``.
    """
    audit_tools = detect_available_audit_tools(folder)
    if not audit_tools:
        raise click.ClickException(
            "Could not detect a supported audit tool. "
            "Expected pnpm-lock, pnpm-lock.yaml and/or uv.lock."
        )

    try:
        for audit_tool in audit_tools:
            run_audit_command(folder=folder, audit_tool=audit_tool)
    except subprocess.CalledProcessError as error:
        raise click.ClickException(
            f"Audit command failed with exit code {error.returncode}."
        ) from error
    except OSError as error:
        raise click.ClickException(str(error)) from error
