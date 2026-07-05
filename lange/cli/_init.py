"""Implementation for the ``lange init`` command."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("init")
def init_command() -> None:
    """Create local ``.lange`` bootstrap files.

    :returns: ``None``.
    """
    lange_dir = Path(".lange")
    lange_dir.mkdir(parents=True, exist_ok=True)
    lange_dir.joinpath(".gitignore").write_text("*\n", encoding="utf-8")
    secrets_file = lange_dir / "secrets.json"
    if not secrets_file.exists():
        secrets_file.write_text("{}\n", encoding="utf-8")
    click.echo("Initialized .lange.")
