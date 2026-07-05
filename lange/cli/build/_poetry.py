"""Poetry-specific build helpers for ``lange build``."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_poetry_build(folder: Path) -> None:
    """
    Run ``poetry build`` in the target folder.

    :param folder: Target folder containing ``pyproject.toml``.
    :returns: ``None``.
    """
    subprocess.run(["poetry", "build"], check=True, cwd=folder)


def run_poetry_version_patch(folder: Path) -> None:
    """
    Run ``poetry version patch`` in the target folder.

    :param folder: Target folder containing ``pyproject.toml``.
    :returns: ``None``.
    """
    subprocess.run(["poetry", "version", "patch"], check=True, cwd=folder)


def run_poetry_publish(folder: Path) -> None:
    """
    Run ``poetry publish`` in the target folder.

    :param folder: Target folder containing ``pyproject.toml``.
    :returns: ``None``.
    """
    subprocess.run(["poetry", "publish"], check=True, cwd=folder)
