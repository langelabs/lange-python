import re
from typing import Final


def is_valid_relay_name(key: str) -> bool:
    """Return whether one mesh name can be used as a public mesh DNS label.

    :param key: Relay name candidate from worker registration.
    :returns: ``True`` when the name is non-empty and DNS-label-safe.
    """
    name_pattern: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9-]+$")
    return name_pattern.fullmatch(key.strip()) is not None