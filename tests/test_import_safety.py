from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_stdlib_types_is_not_shadowed_from_lange_package_directory() -> None:
    """Assert stdlib ``types`` remains importable from inside the ``lange`` package.

    :returns: ``None``.
    """
    package_dir = Path(__file__).resolve().parents[1] / "lange"
    script = (
        "import enum\n"
        "import types\n"
        "print(enum.__file__)\n"
        "print(types.__file__)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    output_paths = result.stdout.splitlines()
    assert len(output_paths) == 2
    assert "lange/types" not in output_paths[1]
    assert output_paths[1].endswith("/types.py")
