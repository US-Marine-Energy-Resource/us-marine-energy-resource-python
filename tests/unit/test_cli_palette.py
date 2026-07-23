"""The CLI color palette: white, blue, and green only, bold only on blue and green.

Two layers of enforcement. A source scan catches off-palette rich markup and
style strings before they ship, and a rendered scan runs real commands with
color forced on and inspects the ANSI codes that actually reach a terminal,
which also catches defaults coming from rich or typer rather than our code.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[2] / "us_marine_energy_resource"

# Files whose output a user sees.
_SOURCE_DIRS = [_PACKAGE / "cli", _PACKAGE / "explore", _PACKAGE / "mer.py"]

_OFF_PALETTE = r"(?:bright_)?(?:red|yellow|magenta|purple|cyan|orange|black)|bright_white"
_FORBIDDEN_MARKUP = re.compile(rf"\[/?(?:bold )?(?:{_OFF_PALETTE})\]")
_FORBIDDEN_STYLE = re.compile(rf'(?:header_)?style="[^"]*(?:{_OFF_PALETTE})[^"]*"')
_BOLD_WHITE = re.compile(r"\[bold\]")


def _source_files() -> list[Path]:
    files: list[Path] = []
    for entry in _SOURCE_DIRS:
        files.extend([entry] if entry.is_file() else sorted(entry.rglob("*.py")))
    return files


def test_sources_use_only_palette_markup() -> None:
    """No off-palette color names or bare bold in any CLI source file."""
    problems = [
        f"{path.relative_to(_PACKAGE)}:{lineno}: {line.strip()}"
        for path in _source_files()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if any(p.search(line) for p in (_FORBIDDEN_MARKUP, _FORBIDDEN_STYLE, _BOLD_WHITE))
    ]
    assert not problems, "off-palette styles found:\n" + "\n".join(problems)


# SGR codes the palette allows: resets, dim, italic, underline, their ends,
# green and blue in both regular and bright forms, and the default colors.
# Bold (1) is checked separately because it is allowed only together with
# green or blue.
_GREEN_OR_BLUE = {32, 34, 92, 94}
_ALLOWED = {0, 1, 2, 3, 4, 22, 23, 24, 39, 49} | _GREEN_OR_BLUE
_SGR = re.compile(r"\x1b\[([0-9;]*)m")


def _violations(text: str) -> list[str]:
    found = []
    for match in _SGR.finditer(text):
        params = match.group(1)
        codes = {int(p) for p in params.split(";") if p}
        if not codes <= _ALLOWED:
            found.append(params)
        elif 1 in codes and not codes & _GREEN_OR_BLUE:
            found.append(params + " (bold without green or blue)")
    return found


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="rich and click strip ANSI codes on Windows pipes, so color never reaches the capture",
)
@pytest.mark.parametrize(
    "args",
    [
        ["--help"],
        ["wave", "--help"],
        ["tidal", "--help"],
        ["explore", "--help"],
        ["sources"],
        ["ls"],
    ],
)
def test_rendered_output_stays_on_palette(args: list[str]) -> None:
    """Real command output, with color forced on, uses only palette codes."""
    env = dict(os.environ)
    env.update({"FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": "100"})
    env.pop("COLORTERM", None)
    env.pop("NO_COLOR", None)
    out = subprocess.run(
        [sys.executable, "-m", "us_marine_energy_resource.mer", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    text = out.stdout + out.stderr
    assert _SGR.search(text), f"expected colored output from {args}"
    bad = _violations(text)
    assert not bad, f"off-palette ANSI codes in `mer {' '.join(args)}`: {sorted(set(bad))}"
