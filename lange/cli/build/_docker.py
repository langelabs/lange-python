"""Docker-specific build helpers for ``lange build``."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import click

BUILDER_NAME = "services-multiarch"
PLATFORMS = "linux/amd64,linux/arm64"
IMAGE_LINE_PATTERN = re.compile(r"^#\s*image:\s*(?P<image>\S+)\s*$", re.IGNORECASE)


def discover_dockerfiles(folder: Path) -> list[Path]:
    """
    Discover Dockerfiles that should be built for a folder.

    :param folder: Folder that should be inspected.
    :returns: Default Dockerfile when present, otherwise sorted named Dockerfiles.
    """
    default_dockerfile = folder / "Dockerfile"
    if default_dockerfile.is_file():
        return [default_dockerfile]

    named_dockerfiles = [
        path
        for path in folder.glob("*.Dockerfile")
        if path.is_file()
    ]
    return sorted(named_dockerfiles, key=lambda item: item.name.lower())


def ensure_docker_is_available() -> None:
    """
    Validate that docker is installed and available in ``PATH``.

    :returns: ``None``.
    """
    if shutil.which("docker") is None:
        raise click.ClickException("docker is not installed or not available in PATH.")


def parse_image_reference(dockerfile_path: Path) -> str:
    """
    Parse and normalize image reference from Dockerfile first line.

    :param dockerfile_path: Dockerfile path to parse.
    :returns: Docker image reference including tag.
    """
    first_line = dockerfile_path.read_text(encoding="utf-8").splitlines()[:1]
    if not first_line:
        raise click.ClickException(
            "Dockerfile must start with '# image: <name>' as first line."
        )

    match = IMAGE_LINE_PATTERN.match(first_line[0].strip())
    if match is None:
        raise click.ClickException(
            "Dockerfile must start with '# image: <name>' as first line."
        )

    image_reference = match.group("image").strip()
    if not image_reference:
        raise click.ClickException(
            "Dockerfile must start with '# image: <name>' as first line."
        )

    if "@" in image_reference or _has_explicit_tag(image_reference):
        return image_reference
    return f"{image_reference}:latest"


def _has_explicit_tag(image_reference: str) -> bool:
    """
    Check whether an image reference already contains a tag component.

    :param image_reference: Docker image reference.
    :returns: ``True`` when the final path segment contains a tag.
    """
    trailing_segment = image_reference.rsplit("/", maxsplit=1)[-1]
    return ":" in trailing_segment


def ensure_buildx_builder() -> None:
    """
    Ensure the configured docker buildx builder exists and is active.

    :returns: ``None``.
    """
    inspect_result = subprocess.run(
        ["docker", "buildx", "inspect", BUILDER_NAME],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if inspect_result.returncode != 0:
        subprocess.run(
            ["docker", "buildx", "create", "--name", BUILDER_NAME, "--use"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        subprocess.run(
            ["docker", "buildx", "use", BUILDER_NAME],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    subprocess.run(
        ["docker", "buildx", "inspect", "--bootstrap"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def run_docker_build(
    folder: Path,
    dockerfile: Path,
    image_reference: str,
    publish: bool,
) -> None:
    """
    Execute docker buildx build for the given folder.

    :param folder: Build context folder.
    :param dockerfile: Dockerfile to build.
    :param image_reference: Docker image name including tag.
    :param publish: Whether ``--push`` should be included.
    :returns: ``None``.
    """
    resolved_dockerfile = dockerfile.resolve()
    ensure_buildx_builder()

    command: list[str] = [
        "docker",
        "buildx",
        "build",
        "--platform",
        PLATFORMS,
        "--file",
        str(resolved_dockerfile),
        "--tag",
        image_reference,
    ]
    if publish:
        command.append("--push")
    command.append(str(folder.resolve()))
    subprocess.run(command, check=True)
