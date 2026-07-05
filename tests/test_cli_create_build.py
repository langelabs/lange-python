"""Tests for ``lange create`` command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from lange.cli import cli


def test_cli_create_adds_service_to_services_json() -> None:
    """Create one service interactively and persist ``services.json``."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["create"],
            input="api\n./services/api\ndocker\nregistry.example.com/api:latest\n",
        )

        assert result.exit_code == 0
        services_file = Path("services.json")
        assert services_file.exists()
        content = services_file.read_text(encoding="utf-8")
        assert '"name": "api"' in content
        assert '"path": "./services/api"' in content
        assert '"build_type": "docker"' in content
        assert '"publish_path": "registry.example.com/api:latest"' in content
