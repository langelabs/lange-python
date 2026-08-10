import platform

from lange.mesh.contracts import PLATFORM_TYPE


def get_platform() -> PLATFORM_TYPE:
    """Return the current platform as a mesh contract value.

    :returns: Normalized operating-system identifier.
    """
    _platform = platform.system()
    if _platform == "Windows":
        return "Windows"
    if _platform == "Linux":
        return "Linux"
    if _platform == "Darwin":
        return "Darwin"
    return "_unknown"
