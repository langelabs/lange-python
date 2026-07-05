"""Tests for the ``lange code audit`` CLI command."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from lange.cli import cli
from lange.cli.code.audit._discovery import detect_available_audit_tools
from lange.cli.code.audit._types import PNPM_AUDIT_TOOL, UV_AUDIT_TOOL


def _completed_process(returncode: int = 0) -> subprocess.CompletedProcess[list[str]]:
    """Create a generic ``CompletedProcess`` result for subprocess mocks.

    :param returncode: Process return code to embed.
    :returns: Completed process value.
    """
    return subprocess.CompletedProcess(args=[], returncode=returncode)


def test_detect_available_audit_tools_returns_supported_tools_in_order(
    tmp_path: Path,
) -> None:
    """Detect supported audit tools based on lock files in one folder."""
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    assert detect_available_audit_tools(tmp_path) == [PNPM_AUDIT_TOOL, UV_AUDIT_TOOL]


def test_cli_code_audit_with_explicit_folder_runs_detected_tool_commands() -> None:
    """Run all detected audit tools for an explicitly selected folder."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        (service_dir / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
        (service_dir / "uv.lock").write_text("version = 1\n", encoding="utf-8")

        with patch("lange.cli.code.audit._runner.subprocess.run") as mocked_run:
            mocked_run.return_value = _completed_process()

            result = runner.invoke(cli, ["code", "audit", "svc"])

        assert result.exit_code == 0
        assert mocked_run.call_count == 2
        assert mocked_run.call_args_list[0].args[0] == ["pnpm", "audit"]
        assert mocked_run.call_args_list[0].kwargs["cwd"] == Path("svc").resolve()
        assert mocked_run.call_args_list[1].args[0] == ["uv", "audit"]
        assert mocked_run.call_args_list[1].kwargs["cwd"] == Path("svc").resolve()


def test_cli_code_audit_without_folder_runs_all_auditable_services() -> None:
    """Run audits across all top-level folders that contain supported lock files."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("alpha").mkdir(parents=True, exist_ok=True)
        Path("beta").mkdir(parents=True, exist_ok=True)
        Path("gamma").mkdir(parents=True, exist_ok=True)
        Path(".hidden").mkdir(parents=True, exist_ok=True)
        (Path("alpha") / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
        (Path("beta") / "uv.lock").write_text("version = 1\n", encoding="utf-8")
        (Path(".hidden") / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")

        with patch("lange.cli.code.audit._command.run_audits_for_folder") as mocked_run:
            result = runner.invoke(cli, ["code", "audit"])

        assert result.exit_code == 0
        assert mocked_run.call_count == 2
        assert [call.kwargs["folder"].name for call in mocked_run.call_args_list] == ["alpha", "beta"]


def test_cli_code_audit_without_folder_errors_when_no_supported_services_exist() -> None:
    """Error when no top-level folder contains a supported audit lock file."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("alpha").mkdir(parents=True, exist_ok=True)
        Path("beta").mkdir(parents=True, exist_ok=True)

        result = runner.invoke(cli, ["code", "audit"])

    assert result.exit_code != 0
    assert "No auditable services were found" in result.output


def test_cli_code_audit_explicit_folder_errors_when_no_supported_lock_file_exists() -> None:
    """Error when the selected folder has no supported audit lock files."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("svc").mkdir(parents=True, exist_ok=True)

        result = runner.invoke(cli, ["code", "audit", "svc"])

    assert result.exit_code != 0
    assert "Could not detect a supported audit tool" in result.output


def test_cli_code_audit_wraps_subprocess_failures_in_click_exception() -> None:
    """Convert subprocess failures into a user-facing click error."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        (service_dir / "uv.lock").write_text("version = 1\n", encoding="utf-8")

        with patch("lange.cli.code.audit._runner.subprocess.run") as mocked_run:
            mocked_run.side_effect = subprocess.CalledProcessError(returncode=2, cmd=["uv", "audit"])

            result = runner.invoke(cli, ["code", "audit", "svc"])

    assert result.exit_code != 0
    assert "Audit command failed with exit code 2." in result.output
