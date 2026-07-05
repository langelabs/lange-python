from pathlib import Path
from typing import Iterable
import os
import click

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    # Python
    ".py",
    # JavaScript / TypeScript / React
    ".js", ".jsx", ".test.ts", ".test.tsx", ".ts", ".tsx",
    # Web (Markup / Styling)
    ".html", ".css", ".scss", ".sass", ".less",
    # Shell / Bash
    ".sh", ".bash", ".zsh",
    # C / C++
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    # Java
    ".java",
    # C#
    ".cs",
    # Ruby
    ".rb",
    # PHP
    ".php",
    # Go
    ".go",
    # Rust
    ".rs",
    # Swift
    ".swift",
    # Kotlin
    ".kt",
    # Dart
    ".dart",
    # Lua
    ".lua",
    # SQL
    ".sql",
    # R
    ".r",
    # Perl
    ".pl",
    # Scala
    ".scala",
)

IGNORED_DIRECTORIES: tuple[str, ...] = (
    ".venv", "venv", "env",
    "node_modules",
    ".git",
    ".next",
    "__pycache__",
    "build",
    "dist",
    "target",
    "vendor",
    ".idea",
    ".vscode"
)


def render_stats_table(stats: dict[str, int], label: str = "File-Type") -> str:
    """
    Render LOC statistics as an ASCII box table.

    :param stats: Mapping from group label to line count.
    :param label: Heading for the first column.
    :returns: Table string with label, LOC and percentage values.
    """
    total = sum(stats.values())
    rows: list[tuple[str, str, str]] = []

    for extension, loc in sorted(stats.items(), key=lambda item: (-item[1], item[0])):
        # Skip extensions with 0 lines to keep the table clean
        if loc == 0:
            continue

        percentage = (loc / total * 100.0) if total else 0.0
        rows.append((extension, str(loc), f"{percentage:.2f}%"))

    total_percentage = 100.0 if total else 0.0
    rows.append(("TOTAL", str(total), f"{total_percentage:.2f}%"))

    headers = (label, "LOC", "Percentage")
    col_widths = [
        max(len(headers[index]), max((len(row[index]) for row in rows), default=0))
        for index in range(3)
    ]

    border = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"

    def _render_row(values: tuple[str, str, str]) -> str:
        padded = [value.ljust(col_widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    lines = [border, _render_row(headers), border]
    lines.extend(_render_row(row) for row in rows)
    lines.append(border)
    return "\n".join(lines)


def _match_supported_extension(file_name: str, extensions: Iterable[str]) -> str | None:
    """
    Resolve the longest supported file ending for a file name.

    :param file_name: File name that should be matched.
    :param extensions: Allowed file endings, including the leading dot.
    :returns: The longest matching ending, or ``None`` when unsupported.
    """
    normalized_file_name = file_name.lower()
    sorted_extensions = sorted(
        (extension.lower() for extension in extensions),
        key=len,
        reverse=True,
    )

    for extension in sorted_extensions:
        if normalized_file_name.endswith(extension):
            return extension

    return None


def count_lines_by_extension(root: Path, extensions: Iterable[str]) -> dict[str, int]:
    """
    Count file lines recursively grouped by file ending.

    :param root: Directory that should be scanned recursively.
    :param extensions: Allowed file endings, including the leading dot.
    :returns: Mapping from file ending to counted LOC.
    """
    normalized_extensions = tuple(extension.lower() for extension in extensions)
    counts = {extension: 0 for extension in normalized_extensions}

    for current_root, directories, files in os.walk(root, topdown=True):
        directories[:] = [name for name in directories if name not in IGNORED_DIRECTORIES]

        for file_name in files:
            suffix = _match_supported_extension(file_name, normalized_extensions)
            if suffix is None:
                continue

            file_path = Path(current_root) / file_name
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
                    counts[suffix] += sum(1 for _ in file_handle)
            except Exception:
                # Silently skip files that can't be opened (e.g., permissions issues)
                pass

    return counts


def count_lines_by_subfolder(root: Path, extensions: Iterable[str]) -> dict[str, int]:
    """
    Count file lines recursively grouped by top-level folder from the root.

    :param root: Directory that should be scanned recursively.
    :param extensions: Allowed file endings, including the leading dot.
    :returns: Mapping from immediate subfolder name to counted LOC.
    """
    counts: dict[str, int] = {}

    for current_root, directories, files in os.walk(root, topdown=True):
        directories[:] = [name for name in directories if name not in IGNORED_DIRECTORIES]

        for file_name in files:
            suffix = _match_supported_extension(file_name, extensions)
            if suffix is None:
                continue

            file_path = Path(current_root) / file_name
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as file_handle:
                    line_count = sum(1 for _ in file_handle)
            except Exception:
                # Silently skip files that can't be opened (e.g., permissions issues)
                continue

            relative_parts = file_path.relative_to(root).parts
            folder_name = relative_parts[0] if len(relative_parts) > 1 else "."
            counts[folder_name] = counts.get(folder_name, 0) + line_count

    return counts


@click.command("stats")
def code_stats() -> None:
    """
    Print LOC statistics for the current working directory.

    :returns: ``None``.
    """
    stats = count_lines_by_extension(Path.cwd(), SUPPORTED_EXTENSIONS)
    subfolder_stats = count_lines_by_subfolder(Path.cwd(), SUPPORTED_EXTENSIONS)

    # Filter out empty languages for the recognized printout so it's not overwhelming
    active_extensions = [ext for ext, count in stats.items() if count > 0]

    click.echo()
    click.echo()
    click.echo(f"Recognized file endings found: {' '.join(active_extensions) if active_extensions else 'None'}")
    click.echo(f"Ignored folders: {' '.join(IGNORED_DIRECTORIES)}")
    click.echo(render_stats_table(stats))
    click.echo()
    click.echo(render_stats_table(subfolder_stats, label="Folder"))


if __name__ == '__main__':
    code_stats()
