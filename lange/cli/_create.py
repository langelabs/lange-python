"""Implementation for the ``lange create`` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click


SERVICES_FILE = Path("services.json")


def _load_services(path: Path) -> list[dict[str, Any]]:
    """Load existing service definitions from disk.

    :param path: JSON file containing service definitions.
    :returns: Existing service list or an empty list when the file is absent.
    """
    if not path.exists():
        return []

    raw_services = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw_services, list):
        return [item for item in raw_services if isinstance(item, dict)]
    if isinstance(raw_services, dict) and isinstance(raw_services.get("services"), list):
        return [
            item for item in raw_services["services"] if isinstance(item, dict)
        ]
    raise click.ClickException("services.json must contain a list of services.")


def _write_services(path: Path, services: list[dict[str, Any]]) -> None:
    """Persist service definitions with stable formatting.

    :param path: JSON file to write.
    :param services: Service definitions to persist.
    """
    path.write_text(
        json.dumps(services, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@click.command("create")
def create_command() -> None:
    """Interactively create one service definition.

    :returns: ``None``.
    """
    service = {
        "name": click.prompt("Service name", type=str),
        "path": click.prompt("Service path", type=str),
        "build_type": click.prompt("Build type", type=str),
        "publish_path": click.prompt("Publish path", type=str),
    }
    services = _load_services(SERVICES_FILE)
    services.append(service)
    _write_services(SERVICES_FILE, services)
    click.echo(f"Created service {service['name']}.")
