"""Tests for the ``lange --version`` CLI option."""

from __future__ import annotations

import re

from click.testing import CliRunner

from lange.cli import cli


def test_cli_version_prints_raw_semver() -> None:
    """Print the installed package version as plain ``X.Y.Z``."""
    runner = CliRunner()

    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert re.fullmatch(r"\d+\.\d+\.\d+\n", result.output) is not None
