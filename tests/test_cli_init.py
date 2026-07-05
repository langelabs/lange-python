"""Tests for the ``lange init`` command."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from lange.cli import cli


def test_cli_init_creates_lange_gitignore() -> None:
    """Create ``.lange`` init files with expected bootstrap content."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["init"])

        assert result.exit_code == 0
        gitignore_file = Path(".lange/.gitignore")
        secrets_file = Path(".lange/secrets.json")
        assert gitignore_file.exists()
        assert secrets_file.exists()
        assert gitignore_file.read_text(encoding="utf-8") == "*\n"
        assert secrets_file.read_text(encoding="utf-8") == "{}\n"


def test_cli_init_preserves_existing_lange_secrets() -> None:
    """Assert rerunning ``lange init`` does not overwrite local secrets."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        lange_dir = Path(".lange")
        lange_dir.mkdir()
        secrets_file = lange_dir / "secrets.json"
        secrets_file.write_text('{"api_key":"secret"}\n', encoding="utf-8")

        result = runner.invoke(cli, ["init"])

        assert result.exit_code == 0
        assert secrets_file.read_text(encoding="utf-8") == '{"api_key":"secret"}\n'
        assert Path(".lange/.gitignore").read_text(encoding="utf-8") == "*\n"
