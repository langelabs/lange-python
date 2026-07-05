import click

from .audit import code_audit
from ._stats import code_stats


@click.group()
def code_group() -> None:
    """
    Group for source-code related commands.

    :returns: ``None``.
    """

code_group.add_command(code_stats, "stats")
code_group.add_command(code_audit, "audit")
