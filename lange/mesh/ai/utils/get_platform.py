from ....contracts.worker import PLATFORM_TYPE
import platform


def get_platform() -> PLATFORM_TYPE:
    _platform = platform.system()
    if _platform == "Windows":
        return "Windows"
    elif _platform == "Linux":
        return "Linux"
    elif _platform == "Darwin":
        return "Darwin"
    else:
        return "_unknown"
