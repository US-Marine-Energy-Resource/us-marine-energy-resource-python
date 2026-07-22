"""``mer`` — the umbrella CLI for marine energy resource data.

Verbs:

- ``mer tidal`` queries the tidal hindcast
- ``mer wave`` queries the wave hindcast
- ``mer sources`` lists the open data sources
- ``mer ls`` / ``info`` / ``explore`` / ``download`` share one path grammar
  (endpoint names, ``s3://`` prefixes, local paths) over h5 / nc / parquet data
- Short aliases: ``list``, ``i``, ``exp``, ``dl``
"""

from __future__ import annotations

import click
import typer
from typer.core import TyperGroup

from .cli._links import DOCS, ISSUES, _link
from .cli.app import _MAIN_HELP, _TIDAL_EPILOG
from .cli.app import main as _tidal_query
from .cli.sources import _SOURCES_HELP
from .cli.sources import sources as _sources
from .cli.wave import _WAVE_EPILOG, _WAVE_HELP
from .cli.wave import wave_query as _wave_query
from .explore.cli import (
    _DOWNLOAD_HELP,
    _EXPLORE_EPILOG,
    _EXPLORE_HELP,
    _INFO_HELP,
    _LS_HELP,
)
from .explore.cli import (
    download as _download,
)
from .explore.cli import (
    explore as _explore,
)
from .explore.cli import (
    info as _info,
)
from .explore.cli import (
    ls as _ls,
)


class _CommandsFirstGroup(TyperGroup):
    """Render the Commands panel above the Options panel in help output."""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Print usage, help text, arguments, commands, options, and epilog."""
        from rich.align import Align
        from rich.padding import Padding
        from typer import rich_utils as ru
        from typer.core import TyperArgument, TyperOption

        if self.rich_markup_mode is None:  # pragma: no cover - rich is always on here
            super().format_help(ctx, formatter)
            return
        markup_mode = self.rich_markup_mode
        console = ru._get_rich_console()

        console.print(Padding(ru.highlighter(self.get_usage(ctx)), 1), style=ru.STYLE_USAGE_COMMAND)
        if self.help:
            help_text = ru._get_help_text(obj=self, markup_mode=markup_mode)
            console.print(Padding(Align(help_text, pad=False), (0, 1, 1, 1)))

        arguments: list[click.Argument] = []
        options: list[click.Option] = []
        for param in self.get_params(ctx):
            if getattr(param, "hidden", False):
                continue
            if isinstance(param, TyperArgument):
                arguments.append(param)
            elif isinstance(param, TyperOption):
                options.append(param)

        ru._print_options_panel(
            name=ru.ARGUMENTS_PANEL_TITLE,
            params=arguments,
            ctx=ctx,
            markup_mode=markup_mode,
            console=console,
        )

        commands = [self.get_command(ctx, name) for name in self.list_commands(ctx)]
        visible = [c for c in commands if c is not None and not c.hidden]
        ru._print_commands_panel(
            name=ru.COMMANDS_PANEL_TITLE,
            commands=visible,
            markup_mode=markup_mode,
            console=console,
            cmd_len=max((len(c.name or "") for c in visible), default=0),
        )

        ru._print_options_panel(
            name=ru.OPTIONS_PANEL_TITLE,
            params=options,
            ctx=ctx,
            markup_mode=markup_mode,
            console=console,
        )

        if self.epilog:
            lines = self.epilog.split("\n\n")
            epilogue = "\n".join(x.replace("\n", " ").strip() for x in lines)
            epilogue_text = ru._make_rich_text(text=epilogue, markup_mode=markup_mode)
            console.print(Padding(Align(epilogue_text, pad=False), 1))


app = typer.Typer(
    name="mer",
    cls=_CommandsFirstGroup,
    help=(
        "U.S. marine energy resource (mer) tools for querying, exploring, and"
        " downloading data from open source data sources (see mer sources).\n\n"
        f"Source Code:      {_link(DOCS)}\n"
        f"Issue Reporting:  {_link(ISSUES)}"
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    add_completion=True,
)

app.command("tidal", help=_MAIN_HELP, epilog=_TIDAL_EPILOG)(_tidal_query)
app.command("wave", help=_WAVE_HELP, epilog=_WAVE_EPILOG)(_wave_query)
app.command("sources", help=_SOURCES_HELP)(_sources)
app.command("ls", help=_LS_HELP)(_ls)
app.command("list", hidden=True)(_ls)
app.command("info", help=_INFO_HELP)(_info)
app.command("i", hidden=True)(_info)
app.command("explore", help=_EXPLORE_HELP, epilog=_EXPLORE_EPILOG)(_explore)
app.command("exp", hidden=True, epilog=_EXPLORE_EPILOG)(_explore)
app.command("download", help=_DOWNLOAD_HELP)(_download)
app.command("dl", hidden=True)(_download)


if __name__ == "__main__":
    app()
