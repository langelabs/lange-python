"""CLI command orchestration for ``lange build``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from ._discovery import (
    detect_available_build_systems,
    list_buildable_folders,
    prompt_for_build_system_selection,
    resolve_build_folder,
)
from ._docker import (
    discover_dockerfiles,
    ensure_docker_is_available,
    parse_image_reference,
    run_docker_build,
)
from ._poetry import run_poetry_build, run_poetry_publish, run_poetry_version_patch
from ._types import DOCKER_BUILD_SYSTEM, POETRY_BUILD_SYSTEM, BuildSystem


@click.command("build")
@click.argument("folder_name", required=False)
@click.option("--push", is_flag=True, help="Publish after a successful build.")
@click.option("--docker", "force_docker", is_flag=True, help="Force docker build.")
@click.option("--poetry", "force_poetry", is_flag=True, help="Force poetry build.")
def build_command(
    folder_name: str | None,
    push: bool,
    force_docker: bool,
    force_poetry: bool,
) -> None:
    """
    Build a folder using docker or poetry with optional publishing.

    :param folder_name: Optional folder name to build.
    :param push: Whether publishing is enabled via flag.
    :param force_docker: Force docker build system.
    :param force_poetry: Force poetry build system.
    :returns: ``None``.
    """
    _validate_force_flags(force_docker=force_docker, force_poetry=force_poetry)

    if folder_name:
        target_folder = resolve_build_folder(folder_name=folder_name, root=Path.cwd())
        _run_build_for_folder(
            folder=target_folder,
            publish=push,
            force_docker=force_docker,
            force_poetry=force_poetry,
            allow_prompt=True,
        )
        return

    target_folders = list_buildable_folders(
        root=Path.cwd(),
        force_docker=force_docker,
        force_poetry=force_poetry,
    )
    if not target_folders:
        raise click.ClickException(
            "No buildable services were found in the current directory."
        )

    for target_folder in target_folders:
        _run_build_for_folder(
            folder=target_folder,
            publish=push,
            force_docker=force_docker,
            force_poetry=force_poetry,
            allow_prompt=False,
        )


def _run_build_for_folder(
    folder: Path,
    publish: bool,
    force_docker: bool,
    force_poetry: bool,
    allow_prompt: bool,
) -> None:
    """
    Resolve build system and run the matching build flow for one folder.

    :param folder: Folder to build.
    :param publish: Whether publishing is enabled via flag.
    :param force_docker: Force docker build system.
    :param force_poetry: Force poetry build system.
    :param allow_prompt: Whether interactive system selection is allowed.
    :returns: ``None``.
    """
    build_system = resolve_build_system(
        folder=folder,
        force_docker=force_docker,
        force_poetry=force_poetry,
        allow_prompt=allow_prompt,
    )

    if build_system == DOCKER_BUILD_SYSTEM:
        _run_docker_flow(folder=folder, publish=publish)
        return

    _run_poetry_flow(folder=folder, publish=publish)


def _validate_force_flags(force_docker: bool, force_poetry: bool) -> None:
    """
    Validate that at most one build-system force flag is provided.

    :param force_docker: Whether docker was forced.
    :param force_poetry: Whether poetry was forced.
    :returns: ``None``.
    """
    if force_docker and force_poetry:
        raise click.ClickException("Only one of --docker or --poetry can be used.")


def resolve_build_system(
    folder: Path,
    force_docker: bool,
    force_poetry: bool,
    allow_prompt: bool = True,
) -> BuildSystem:
    """
    Resolve the build system from force flags or discovered files.

    :param folder: Folder that should be built.
    :param force_docker: Whether docker was forced.
    :param force_poetry: Whether poetry was forced.
    :param allow_prompt: Whether selection prompts are allowed for ambiguity.
    :returns: Resolved build system value.
    """
    if force_docker:
        if not discover_dockerfiles(folder):
            raise click.ClickException(
                f"Dockerfile was not found at '{folder / 'Dockerfile'}'."
            )
        return DOCKER_BUILD_SYSTEM

    if force_poetry:
        pyproject_file = folder / "pyproject.toml"
        if not pyproject_file.is_file():
            raise click.ClickException(
                f"pyproject.toml was not found at '{pyproject_file}'."
            )
        return POETRY_BUILD_SYSTEM

    detected_systems = detect_available_build_systems(folder)
    if not detected_systems:
        raise click.ClickException(
            "Could not detect a supported build system. "
            "Expected Dockerfile and/or pyproject.toml."
        )
    if len(detected_systems) == 1:
        return detected_systems[0]
    if not allow_prompt:
        return detected_systems[0]
    return prompt_for_build_system_selection(detected_systems)


def _run_docker_flow(folder: Path, publish: bool) -> None:
    """
    Run docker build flow and optional publish flow.

    :param folder: Folder to build.
    :param publish: Whether publish was requested via flag.
    :returns: ``None``.
    """
    dockerfiles = discover_dockerfiles(folder)
    if not dockerfiles:
        raise click.ClickException(
            f"Dockerfile was not found at '{folder / 'Dockerfile'}'."
        )

    ensure_docker_is_available()

    try:
        for dockerfile in dockerfiles:
            image_reference = parse_image_reference(dockerfile)
            run_docker_build(
                folder=folder,
                dockerfile=dockerfile,
                image_reference=image_reference,
                publish=publish,
            )
            if not publish and _confirm_publish():
                run_docker_build(
                    folder=folder,
                    dockerfile=dockerfile,
                    image_reference=image_reference,
                    publish=True,
                )
    except subprocess.CalledProcessError as error:
        raise click.ClickException(
            f"Docker command failed with exit code {error.returncode}."
        ) from error
    except OSError as error:
        raise click.ClickException(str(error)) from error


def _run_poetry_flow(folder: Path, publish: bool) -> None:
    """
    Run poetry build flow and optional publish flow.

    :param folder: Folder to build.
    :param publish: Whether publish was requested via flag.
    :returns: ``None``.
    """
    try:
        if publish:
            run_poetry_version_patch(folder=folder)
        run_poetry_build(folder=folder)
        if publish or _confirm_publish():
            run_poetry_publish(folder=folder)
    except subprocess.CalledProcessError as error:
        raise click.ClickException(
            f"Poetry command failed with exit code {error.returncode}."
        ) from error
    except OSError as error:
        raise click.ClickException(str(error)) from error


def _confirm_publish() -> bool:
    """
    Ask whether the built artifact should be published.

    :returns: ``True`` when publish was confirmed.
    """
    return click.confirm("Build finished. Publish now?", default=False)
