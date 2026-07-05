"""CLI commands for the ``lange`` Python package."""

from __future__ import annotations

import click

from .build import build_command
from .code import code_group
from ._create import create_command
from ._init import init_command



@click.group()
@click.version_option(package_name="lange-python", message="%(version)s")
def cli() -> None:
    """
    Lange CLI entrypoint.

    :returns: ``None``.
    """

cli.add_command(code_group, "code")
cli.add_command(build_command, "build")
cli.add_command(create_command, "create")
cli.add_command(init_command, "init")
