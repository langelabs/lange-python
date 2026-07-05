"""Tests for the ``lange build`` CLI command."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from lange.cli import cli


def _completed_process(returncode: int = 0) -> subprocess.CompletedProcess[list[str]]:
    """Create a generic successful ``CompletedProcess`` result.

    :param returncode: Process return code to embed.
    :returns: Completed process value.
    """
    return subprocess.CompletedProcess(args=[], returncode=returncode)


def _write_dockerfile(path: Path, first_line: str) -> None:
    """Write a Dockerfile with a first-line image marker.

    :param path: File path that should receive Dockerfile content.
    :param first_line: First line of the Dockerfile.
    :returns: ``None``.
    """
    path.write_text(f"{first_line}\nFROM scratch\n", encoding="utf-8")


def test_cli_build_with_explicit_folder_uses_docker_and_latest_tag() -> None:
    """Build from a provided folder and append ``:latest`` when no tag is set."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        _write_dockerfile(service_dir / "Dockerfile", "# image: registry.local/svc")

        with (
            patch("lange.cli.build._docker.shutil.which", return_value="/usr/bin/docker"),
            patch("lange.cli.build._docker.subprocess.run") as mocked_run,
        ):
            mocked_run.side_effect = [
                _completed_process(returncode=1),
                _completed_process(),
                _completed_process(),
                _completed_process(),
            ]

            result = runner.invoke(cli, ["build", "svc"], input="n\n")

        assert result.exit_code == 0
        command = mocked_run.call_args_list[-1].args[0]
        assert command[:4] == ["docker", "buildx", "build", "--platform"]
        assert "--tag" in command
        assert "registry.local/svc:latest" in command
        assert "--push" not in command


def test_cli_build_without_folder_builds_all_buildable_services() -> None:
    """Build every top-level non-hidden service that has a supported build file."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path(".hidden").mkdir(parents=True, exist_ok=True)
        Path("alpha").mkdir(parents=True, exist_ok=True)
        Path("beta").mkdir(parents=True, exist_ok=True)
        Path("gamma").mkdir(parents=True, exist_ok=True)
        Path("alpha/pyproject.toml").write_text("[tool.poetry]\n", encoding="utf-8")
        _write_dockerfile(Path("beta/Dockerfile"), "# image: registry.local/beta")
        Path(".hidden/pyproject.toml").write_text("[tool.poetry]\n", encoding="utf-8")

        with (
            patch("lange.cli.build._command._run_poetry_flow") as mocked_poetry_flow,
            patch("lange.cli.build._command._run_docker_flow") as mocked_docker_flow,
        ):
            result = runner.invoke(cli, ["build"])

        assert result.exit_code == 0
        assert "Choose a folder to build:" not in result.output
        assert mocked_poetry_flow.call_count == 1
        assert mocked_poetry_flow.call_args_list[0].kwargs["folder"].name == "alpha"
        assert mocked_docker_flow.call_count == 1
        assert mocked_docker_flow.call_args_list[0].kwargs["folder"].name == "beta"


def test_cli_build_without_folder_detects_named_dockerfiles() -> None:
    """Build top-level services that only contain named Dockerfiles."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("api").mkdir(parents=True, exist_ok=True)
        _write_dockerfile(Path("api/api.Dockerfile"), "# image: registry.local/api")
        _write_dockerfile(Path("api/worker.Dockerfile"), "# image: registry.local/worker")

        with (
            patch("lange.cli.build._command._run_poetry_flow") as mocked_poetry_flow,
            patch("lange.cli.build._command._run_docker_flow") as mocked_docker_flow,
        ):
            result = runner.invoke(cli, ["build"])

        assert result.exit_code == 0
        assert mocked_poetry_flow.call_count == 0
        assert mocked_docker_flow.call_count == 1
        assert mocked_docker_flow.call_args_list[0].kwargs["folder"].name == "api"


def test_cli_build_without_folder_prefers_named_dockerfiles_over_pyproject() -> None:
    """Build API-shaped services with named Dockerfiles using docker."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("api").mkdir(parents=True, exist_ok=True)
        Path("api/pyproject.toml").write_text("[project]\nname = \"api\"\n", encoding="utf-8")
        _write_dockerfile(Path("api/api.Dockerfile"), "# image: registry.local/api")
        _write_dockerfile(Path("api/worker.Dockerfile"), "# image: registry.local/worker")

        with (
            patch("lange.cli.build._command._run_poetry_flow") as mocked_poetry_flow,
            patch("lange.cli.build._command._run_docker_flow") as mocked_docker_flow,
        ):
            result = runner.invoke(cli, ["build", "--push"])

        assert result.exit_code == 0
        assert mocked_poetry_flow.call_count == 0
        assert mocked_docker_flow.call_count == 1
        assert mocked_docker_flow.call_args_list[0].kwargs["folder"].name == "api"
        assert mocked_docker_flow.call_args_list[0].kwargs["publish"] is True


def test_cli_build_with_named_dockerfiles_builds_each_file() -> None:
    """Build every ``*.Dockerfile`` when no default Dockerfile exists."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("api")
        service_dir.mkdir(parents=True, exist_ok=True)
        _write_dockerfile(service_dir / "api.Dockerfile", "# image: registry.local/api")
        _write_dockerfile(
            service_dir / "worker.Dockerfile",
            "# image: registry.local/worker",
        )

        with (
            patch("lange.cli.build._docker.shutil.which", return_value="/usr/bin/docker"),
            patch("lange.cli.build._docker.subprocess.run") as mocked_run,
        ):
            mocked_run.side_effect = [
                _completed_process(returncode=1),
                _completed_process(),
                _completed_process(),
                _completed_process(),
                _completed_process(),
                _completed_process(),
                _completed_process(),
                _completed_process(),
            ]
            result = runner.invoke(cli, ["build", "api", "--docker"], input="n\nn\n")

        assert result.exit_code == 0
        build_commands = [
            call.args[0]
            for call in mocked_run.call_args_list
            if call.args[0][:3] == ["docker", "buildx", "build"]
        ]
        assert len(build_commands) == 2
        assert any("api.Dockerfile" in command for command in build_commands[0])
        assert "registry.local/api:latest" in build_commands[0]
        assert any("worker.Dockerfile" in command for command in build_commands[1])
        assert "registry.local/worker:latest" in build_commands[1]


def test_cli_build_prefers_default_dockerfile_over_named_dockerfiles() -> None:
    """Build only ``Dockerfile`` when default and named Dockerfiles coexist."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        _write_dockerfile(service_dir / "Dockerfile", "# image: registry.local/svc")
        _write_dockerfile(service_dir / "worker.Dockerfile", "# image: registry.local/worker")

        with (
            patch("lange.cli.build._docker.shutil.which", return_value="/usr/bin/docker"),
            patch("lange.cli.build._docker.subprocess.run") as mocked_run,
        ):
            mocked_run.side_effect = [
                _completed_process(returncode=1),
                _completed_process(),
                _completed_process(),
                _completed_process(),
            ]
            result = runner.invoke(cli, ["build", "svc", "--docker"], input="n\n")

        assert result.exit_code == 0
        build_commands = [
            call.args[0]
            for call in mocked_run.call_args_list
            if call.args[0][:3] == ["docker", "buildx", "build"]
        ]
        assert len(build_commands) == 1
        assert any(command.endswith("Dockerfile") for command in build_commands[0])
        assert not any("worker.Dockerfile" in command for command in build_commands[0])


def test_cli_build_without_folder_and_force_poetry_builds_all_poetry_services() -> None:
    """Build only folders containing ``pyproject.toml`` when ``--poetry`` is set."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("alpha").mkdir(parents=True, exist_ok=True)
        Path("beta").mkdir(parents=True, exist_ok=True)
        Path("alpha/pyproject.toml").write_text("[tool.poetry]\n", encoding="utf-8")
        _write_dockerfile(Path("beta/Dockerfile"), "# image: registry.local/beta")

        with (
            patch("lange.cli.build._command._run_poetry_flow") as mocked_poetry_flow,
            patch("lange.cli.build._command._run_docker_flow") as mocked_docker_flow,
        ):
            result = runner.invoke(cli, ["build", "--poetry"])

        assert result.exit_code == 0
        assert mocked_poetry_flow.call_count == 1
        assert mocked_poetry_flow.call_args_list[0].kwargs["folder"].name == "alpha"
        assert mocked_docker_flow.call_count == 0


def test_cli_build_without_folder_errors_when_no_buildable_services_exist() -> None:
    """Error when no top-level service folder has supported build files."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("alpha").mkdir(parents=True, exist_ok=True)
        Path("beta").mkdir(parents=True, exist_ok=True)
        result = runner.invoke(cli, ["build"])

    assert result.exit_code != 0
    assert "No buildable services were found" in result.output


def test_cli_build_errors_when_both_force_flags_are_passed() -> None:
    """Reject force-flag conflicts."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("svc").mkdir(parents=True, exist_ok=True)
        result = runner.invoke(cli, ["build", "svc", "--docker", "--poetry"])

    assert result.exit_code != 0
    assert "Only one of --docker or --poetry can be used" in result.output


def test_cli_build_force_docker_requires_dockerfile() -> None:
    """Error when ``--docker`` is passed but Dockerfile is missing."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("svc").mkdir(parents=True, exist_ok=True)
        result = runner.invoke(cli, ["build", "svc", "--docker"])

    assert result.exit_code != 0
    assert "Dockerfile was not found" in result.output


def test_cli_build_force_poetry_requires_pyproject() -> None:
    """Error when ``--poetry`` is passed but ``pyproject.toml`` is missing."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("svc").mkdir(parents=True, exist_ok=True)
        result = runner.invoke(cli, ["build", "svc", "--poetry"])

    assert result.exit_code != 0
    assert "pyproject.toml was not found" in result.output


def test_cli_build_auto_detect_with_both_prompts_for_system_choice() -> None:
    """Prompt for build system choice when both Docker and Poetry are available."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        _write_dockerfile(service_dir / "Dockerfile", "# image: registry.local/svc")
        (service_dir / "pyproject.toml").write_text("[tool.poetry]\n", encoding="utf-8")

        with patch("lange.cli.build._poetry.subprocess.run") as mocked_run:
            mocked_run.return_value = _completed_process()
            result = runner.invoke(cli, ["build", "svc"], input="2\nn\n")

        assert result.exit_code == 0
        assert "1. docker" in result.output
        assert "2. poetry" in result.output
        command = mocked_run.call_args_list[0].args[0]
        assert command == ["poetry", "build"]


def test_cli_build_auto_detect_none_errors() -> None:
    """Error when no supported build files are available."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("svc").mkdir(parents=True, exist_ok=True)
        result = runner.invoke(cli, ["build", "svc"])

    assert result.exit_code != 0
    assert "Could not detect a supported build system" in result.output


def test_cli_build_invalid_docker_image_comment_errors() -> None:
    """Reject Dockerfiles without the required first-line image marker."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        (service_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

        result = runner.invoke(cli, ["build", "svc", "--docker"])

    assert result.exit_code != 0
    assert "must start with '# image: <name>'" in result.output


def test_cli_build_with_publish_flag_pushes_for_docker() -> None:
    """Build with ``--push`` should pass ``--push`` to docker buildx."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        _write_dockerfile(service_dir / "Dockerfile", "# image: registry.local/svc")

        with (
            patch("lange.cli.build._docker.shutil.which", return_value="/usr/bin/docker"),
            patch("lange.cli.build._docker.subprocess.run") as mocked_run,
        ):
            mocked_run.side_effect = [
                _completed_process(returncode=1),
                _completed_process(),
                _completed_process(),
                _completed_process(),
            ]
            result = runner.invoke(cli, ["build", "svc", "--docker", "--push"])

        assert result.exit_code == 0
        command = mocked_run.call_args_list[-1].args[0]
        assert "--push" in command


def test_cli_build_poetry_prompt_publish_runs_publish_after_build() -> None:
    """When publish is confirmed interactively, run poetry publish after build."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        (service_dir / "pyproject.toml").write_text("[tool.poetry]\n", encoding="utf-8")

        with patch("lange.cli.build._poetry.subprocess.run") as mocked_run:
            mocked_run.return_value = _completed_process()
            result = runner.invoke(cli, ["build", "svc", "--poetry"], input="y\n")

        assert result.exit_code == 0
        assert mocked_run.call_args_list[0].args[0] == ["poetry", "build"]
        assert mocked_run.call_args_list[1].args[0] == ["poetry", "publish"]


def test_cli_build_poetry_push_bumps_patch_before_build() -> None:
    """When ``--push`` is set for poetry, bump the patch version before build."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        (service_dir / "pyproject.toml").write_text("[tool.poetry]\n", encoding="utf-8")

        with patch("lange.cli.build._poetry.subprocess.run") as mocked_run:
            mocked_run.return_value = _completed_process()
            result = runner.invoke(cli, ["build", "svc", "--poetry", "--push"])

        assert result.exit_code == 0
        assert mocked_run.call_args_list[0].args[0] == ["poetry", "version", "patch"]
        assert mocked_run.call_args_list[1].args[0] == ["poetry", "build"]
        assert mocked_run.call_args_list[2].args[0] == ["poetry", "publish"]


def test_cli_build_docker_respects_existing_tag() -> None:
    """Preserve image tags provided in the Dockerfile image marker."""
    runner = CliRunner()

    with runner.isolated_filesystem():
        service_dir = Path("svc")
        service_dir.mkdir(parents=True, exist_ok=True)
        _write_dockerfile(service_dir / "Dockerfile", "# image: registry.local/svc:1.2.3")

        with (
            patch("lange.cli.build._docker.shutil.which", return_value="/usr/bin/docker"),
            patch("lange.cli.build._docker.subprocess.run") as mocked_run,
        ):
            mocked_run.side_effect = [
                _completed_process(returncode=1),
                _completed_process(),
                _completed_process(),
                _completed_process(),
            ]
            result = runner.invoke(cli, ["build", "svc", "--docker"], input="n\n")

        assert result.exit_code == 0
        command = mocked_run.call_args_list[-1].args[0]
        assert "registry.local/svc:1.2.3" in command
