"""CLI routing for ``mer`` and ``mer explore``, plus the lazy-import contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from us_marine_energy_resource.mer import app

runner = CliRunner()


def test_lazy_import_contract() -> None:
    """Importing the package, wave included, must not pull in heavy dependencies."""
    code = (
        "import sys\n"
        "import us_marine_energy_resource\n"
        "import us_marine_energy_resource.wave_hindcast\n"
        "heavy = ('h5py', 'boto3', 'pyarrow', 'duckdb', 'pooch', 'pandas')\n"
        "print([m for m in heavy if m in sys.modules])\n"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "[]"


def test_mer_help_lists_domains() -> None:
    """Mer help lists domains."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tidal" in result.output
    assert "wave" in result.output
    assert "sources" in result.output
    assert "explore" in result.output


def test_mer_help_commands_before_options() -> None:
    """The Commands panel renders above the Options panel."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert result.output.index("Commands") < result.output.index("Options")


def test_mer_sources_lists_both_datasets() -> None:
    """Mer sources lists each dataset's submission, browser view, and bucket."""
    result = runner.invoke(app, ["sources"])
    assert result.exit_code == 0
    assert "mhkdr.openei.org/submissions/632" in result.output
    assert "mhkdr.openei.org/submissions/326" in result.output
    assert "s3://marine-energy-data/us-tidal/" in result.output
    assert "s3://wpto-pds-us-wave/" in result.output


def test_mer_explore_help_lists_modes() -> None:
    """Mer explore help lists the mode flags."""
    result = runner.invoke(app, ["explore", "--help"])
    for flag in ("--info", "--tree", "--attrs", "--head", "--stats"):
        assert flag in result.output


def test_mer_tidal_binds_positional() -> None:
    """Mer tidal binds positional."""
    # Conflicting geometry proves the coordinate bound to the callback, no network.
    result = runner.invoke(app, ["tidal", "60.73,-151.43", "--bbox", "1,2,3,4"])
    assert result.exit_code == 1
    assert "Multiple geometry" in result.output


def test_explore_path_first_overview(h5_file: Path) -> None:
    """A bare path prints the overview (header + tree)."""
    result = runner.invoke(app, ["explore", str(h5_file)])
    assert result.exit_code == 0
    assert "format" in result.output
    assert "significant_wave_height" in result.output


def test_explore_info_json(parquet_file: Path) -> None:
    """Path-first --info --json emits parseable JSON."""
    result = runner.invoke(app, ["explore", str(parquet_file), "--info", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output)["format"] == "parquet"


def test_explore_tree_runs(h5_file: Path) -> None:
    """Path-first --tree --storage runs."""
    result = runner.invoke(app, ["explore", str(h5_file), "--tree", "--storage"])
    assert result.exit_code == 0
    assert "significant_wave_height" in result.output


def test_explore_no_args_shows_overview() -> None:
    """A bare `mer explore` prints the overview and exits 0."""
    result = runner.invoke(app, ["explore"])
    assert result.exit_code == 0
    assert "Browse a dataset" in result.output


def test_ls_and_info_overviews_name_their_own_verb() -> None:
    """Bare `mer ls` and `mer info` talk about themselves, not about explore."""
    for verb in ("ls", "info"):
        result = runner.invoke(app, [verb])
        assert result.exit_code == 0
        assert f"mer {verb} tidal" in result.output
        assert f"Run mer {verb} --help" in result.output
        assert "mer explore" not in result.output


def test_explore_two_modes_rejected(h5_file: Path) -> None:
    """Passing two mode flags is rejected."""
    result = runner.invoke(app, ["explore", str(h5_file), "--tree", "--info"])
    assert result.exit_code == 1
    assert "one mode" in result.output


def test_head_rejects_n_and_index(h5_file: Path) -> None:
    """Head rejects n and index."""
    result = runner.invoke(
        app,
        [
            "explore",
            str(h5_file),
            "--head",
            "-p",
            "/significant_wave_height",
            "-n",
            "3",
            "--index",
            "0:3",
        ],
    )
    assert result.exit_code == 1
    assert "not both" in result.output


def test_head_requires_path(h5_file: Path) -> None:
    """Head requires path."""
    result = runner.invoke(app, ["explore", str(h5_file), "--head"])
    assert result.exit_code == 1
    assert "requires --path" in result.output


def test_dry_run_exits_zero(h5_file: Path) -> None:
    """Dry run exits zero."""
    result = runner.invoke(
        app,
        ["explore", str(h5_file), "--stats", "-p", "/significant_wave_height", "--dry-run"],
    )
    assert result.exit_code == 0
    assert "plan" in result.output


def test_exact_over_budget_is_refused(h5_file: Path) -> None:
    """Exact over budget is refused."""
    result = runner.invoke(
        app,
        [
            "explore",
            str(h5_file),
            "--stats",
            "-p",
            "/significant_wave_height",
            "--exact",
            "--max-memory-mb",
            "0.000001",
        ],
    )
    assert result.exit_code == 1
    assert "Refused" in result.output
