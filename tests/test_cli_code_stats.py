"""Tests for the ``lange code stats`` CLI command and helpers."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from lange.cli import cli
from lange.cli.code._stats import (
    SUPPORTED_EXTENSIONS,
    count_lines_by_extension,
    count_lines_by_subfolder,
    render_stats_table,
)


def _write_file(path: Path, content: str) -> None:
    """Create a UTF-8 text file including parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_count_lines_by_extension_recursively(tmp_path: Path) -> None:
    """Count line totals recursively and ignore unsupported file endings."""
    _write_file(tmp_path / "main.py", "print('a')\nprint('b')\n")
    _write_file(tmp_path / "web" / "component.tsx", "a\nb\nc\n")
    _write_file(tmp_path / "assets" / "styles.css", "x\n")
    _write_file(tmp_path / "notes.md", "ignored\n")

    stats = count_lines_by_extension(tmp_path, SUPPORTED_EXTENSIONS)

    assert stats[".py"] == 2
    assert stats[".tsx"] == 3
    assert stats[".css"] == 1
    assert stats[".js"] == 0
    assert stats[".ts"] == 0


def test_count_lines_by_extension_prefers_longest_supported_suffix(tmp_path: Path) -> None:
    """Count files against the longest matching ending before shorter suffixes."""
    _write_file(tmp_path / "src" / "alpha.test.tsx", "1\n2\n")
    _write_file(tmp_path / "src" / "beta.test.ts", "1\n")
    _write_file(tmp_path / "src" / "gamma.test.test.tsx", "1\n2\n3\n")
    _write_file(tmp_path / "src" / "delta.tsx", "1\n")

    stats = count_lines_by_extension(tmp_path, SUPPORTED_EXTENSIONS)

    assert stats[".test.tsx"] == 5
    assert stats[".test.ts"] == 1
    assert stats[".tsx"] == 1
    assert stats[".ts"] == 0


def test_count_lines_by_extension_skips_ignored_directories(tmp_path: Path) -> None:
    """Skip expensive or irrelevant directories during recursive scanning."""
    _write_file(tmp_path / "src" / "main.py", "1\n2\n")
    _write_file(tmp_path / "node_modules" / "dep.py", "1\n2\n3\n4\n")
    _write_file(tmp_path / ".git" / "hooks" / "ignored.py", "1\n2\n3\n")
    _write_file(tmp_path / ".venv" / "lib.py", "1\n")
    _write_file(tmp_path / ".next" / "cache.py", "1\n2\n3\n")

    stats = count_lines_by_extension(tmp_path, SUPPORTED_EXTENSIONS)

    assert stats[".py"] == 2


def test_count_lines_by_subfolder_groups_lines_relative_to_root(tmp_path: Path) -> None:
    """Aggregate LOC totals by immediate subfolder from the invocation directory."""
    _write_file(tmp_path / "main.py", "1\n")
    _write_file(tmp_path / "backend" / "api.py", "1\n2\n")
    _write_file(tmp_path / "backend" / "routes" / "users.ts", "1\n2\n3\n")
    _write_file(tmp_path / "frontend" / "app.tsx", "1\n2\n")
    _write_file(tmp_path / "frontend" / "spec.test.tsx", "1\n")
    _write_file(tmp_path / "notes.md", "ignored\n")

    stats = count_lines_by_subfolder(tmp_path, SUPPORTED_EXTENSIONS)

    assert stats["."] == 1
    assert stats["backend"] == 5
    assert stats["frontend"] == 3


def test_render_stats_table_includes_total_and_percentages() -> None:
    """Render an ASCII table with totals and percentages."""
    stats = {extension: 0 for extension in SUPPORTED_EXTENSIONS}
    stats[".py"] = 3
    stats[".js"] = 1

    table = render_stats_table(stats)

    assert table.startswith("+")
    assert "| File-Type " in table
    assert "| .py " in table
    assert "75.00%" in table
    assert "| TOTAL " in table
    assert "| 4 " in table
    assert "100.00%" in table


def test_render_stats_table_orders_file_types_by_loc_desc() -> None:
    """Sort extension rows in descending LOC order."""
    stats = {extension: 0 for extension in SUPPORTED_EXTENSIONS}
    stats[".js"] = 4
    stats[".py"] = 1
    stats[".tsx"] = 2

    table = render_stats_table(stats)
    lines = table.splitlines()
    row_lines = [line for line in lines if line.startswith("| .")]

    assert row_lines[0].startswith("| .js")
    assert row_lines[1].startswith("| .tsx")
    assert row_lines[2].startswith("| .py")


def test_cli_code_stats_command_outputs_recognized_endings_and_table() -> None:
    """Run the CLI command and assert expected output shape."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        _write_file(Path("src") / "one.py", "1\n2\n")
        _write_file(Path("src") / "two.js", "1\n")
        _write_file(Path("web") / "three.test.tsx", "1\n2\n")
        _write_file(Path("README.md"), "ignored\n")

        result = runner.invoke(cli, ["code", "stats"])

    assert result.exit_code == 0
    assert result.output.startswith("\n\n")
    assert "Recognized file endings found: .py .js .test.tsx" in result.output
    assert "Ignored folders: .venv venv env node_modules .git .next __pycache__ build dist target vendor .idea .vscode" in result.output
    assert "| .py " in result.output
    assert "| .js " in result.output
    assert "| .test.tsx " in result.output
    assert "| TOTAL " in result.output
    assert "| Folder " in result.output
    assert "| src " in result.output
    assert "| web " in result.output
    assert "| 5 " in result.output
