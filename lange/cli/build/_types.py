"""Shared build command type definitions."""

from __future__ import annotations

from typing import Literal

BuildSystem = Literal["docker", "poetry"]

DOCKER_BUILD_SYSTEM: BuildSystem = "docker"
POETRY_BUILD_SYSTEM: BuildSystem = "poetry"
