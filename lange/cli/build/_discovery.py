"""Folder and build-system discovery helpers for ``lange build``."""

from __future__ import annotations

from pathlib import Path

import click

from ._docker import discover_dockerfiles
from ._types import DOCKER_BUILD_SYSTEM, POETRY_BUILD_SYSTEM, BuildSystem


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


def prompt_for_folder_selection(folders: list[Path]) -> Path:
    """
    Prompt the user to choose one folder from candidates.

    :param folders: Selectable folders.
    :returns: Selected folder path.
    """
    if not folders:
        raise click.ClickException(
            "No selectable folders were found in the current directory."
        )

    click.echo("Choose a folder to build:")
    for index, folder in enumerate(folders, start=1):
        click.echo(f"{index}. {folder.name}")

    selection = click.prompt(
        "Selection",
        type=click.IntRange(min=1, max=len(folders)),
    )
    return folders[selection - 1]


def resolve_build_folder(folder_name: str | None, root: Path) -> Path:
    """
    Resolve the build folder from argument value or interactive selection.

    :param folder_name: Optional folder argument passed via CLI.
    :param root: Current working directory.
    :returns: Target folder for the build.
    """
    if folder_name:
        folder = (root / folder_name).resolve()
        if not folder.exists() or not folder.is_dir():
            raise click.ClickException(
                f"Folder '{folder_name}' was not found or is not a directory."
            )
        return folder

    folders = list_candidate_folders(root)
    return prompt_for_folder_selection(folders).resolve()


def list_buildable_folders(
    root: Path,
    force_docker: bool,
    force_poetry: bool,
) -> list[Path]:
    """
    List top-level folders that are buildable for the selected mode.

    :param root: Directory that should be scanned.
    :param force_docker: Whether docker mode was explicitly forced.
    :param force_poetry: Whether poetry mode was explicitly forced.
    :returns: Sorted list of buildable folders.
    """
    candidates = list_candidate_folders(root)
    if force_docker:
        return [folder for folder in candidates if discover_dockerfiles(folder)]
    if force_poetry:
        return [folder for folder in candidates if (folder / "pyproject.toml").is_file()]
    return [
        folder for folder in candidates if detect_available_build_systems(folder)
    ]


def detect_available_build_systems(folder: Path) -> list[BuildSystem]:
    """
    Detect supported build systems available in the given folder.

    :param folder: Folder that should be inspected.
    :returns: Ordered list of detected build systems.
    """
    available: list[BuildSystem] = []
    if discover_dockerfiles(folder):
        available.append(DOCKER_BUILD_SYSTEM)
    if (folder / "pyproject.toml").is_file():
        available.append(POETRY_BUILD_SYSTEM)
    return available


def prompt_for_build_system_selection(
    available_systems: list[BuildSystem],
) -> BuildSystem:
    """
    Prompt user for a build-system selection.

    :param available_systems: Selectable build systems.
    :returns: Selected build system.
    """
    if not available_systems:
        raise click.ClickException("No build systems were provided for selection.")

    click.echo("Multiple build systems detected. Choose one:")
    for index, system in enumerate(available_systems, start=1):
        click.echo(f"{index}. {system}")

    selection = click.prompt(
        "Selection",
        type=click.IntRange(min=1, max=len(available_systems)),
    )
    return available_systems[selection - 1]
