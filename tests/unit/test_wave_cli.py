"""The ``mer wave`` command, through the umbrella app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from us_marine_energy_resource.mer import app

runner = CliRunner()

PACWAVE_ARG = "44.5670485,-124.22896475"

_DESCRIBE = {
    "location_id": 479519,
    "domain": "West_Coast",
    "endpoint": "us-west-coast-hindcast-download",
    "requested_lat": 44.567,
    "requested_lon": -124.229,
    "node_lat": 44.5682,
    "node_lon": -124.228,
    "distance_m": 142.3,
    "years": [1979, 2020],
    "n_years": 42,
    "interval_minutes": 180,
    "direction_transform": None,
}


@pytest.fixture
def described(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub node resolution and the variable probe so nothing touches the network."""
    monkeypatch.setattr(
        "us_marine_energy_resource.cli.wave.hindcast.describe_point",
        lambda lat, lon, **kwargs: dict(_DESCRIBE),
    )
    monkeypatch.setattr(
        "us_marine_energy_resource.wave_hindcast.nlr_api.client.attributes_for",
        lambda domain, api_key, email, on_event=None: [
            "significant_wave_height",
            "energy_period",
        ],
    )
    return _DESCRIBE


def test_wave_registered_on_mer() -> None:
    """The umbrella help lists the wave verb."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "wave" in result.output


def test_no_geometry_errors() -> None:
    """No point is a usage error."""
    result = runner.invoke(app, ["wave"])
    assert result.exit_code == 1
    assert "lat,lon" in result.output


def test_bad_point_errors() -> None:
    """An unparseable coordinate is a clean error."""
    result = runner.invoke(app, ["wave", "not-a-point"])
    assert result.exit_code == 1


def test_info_needs_no_credentials(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--info renders the node description and exits cleanly, key or no key."""
    monkeypatch.delenv("NLR_DEVELOPER_API_KEY", raising=False)
    monkeypatch.delenv("NLR_DEVELOPER_EMAIL", raising=False)
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--info"])
    assert result.exit_code == 0
    assert "West_Coast" in result.output
    assert "479519" in result.output
    assert "1979-2020" in result.output


def test_dry_run_sends_nothing(described: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """--dry-run shows the would-be request and never calls the fetch path."""

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry run must not fetch")

    monkeypatch.setattr("us_marine_energy_resource.cli.wave.hindcast.get_data_at_point", boom)
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--dry-run"])
    assert result.exit_code == 0
    assert "Would request" in result.output


def test_missing_credentials_fail_before_confirm(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit api fetch without credentials errors up front."""
    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NLR_DEVELOPER_API_KEY", raising=False)
    monkeypatch.delenv("NLR_DEVELOPER_EMAIL", raising=False)
    # Keep the developer's own .env / ~/.mer.env out of the test.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: empty))
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--backend", "api", "--yes"])
    assert result.exit_code == 1
    assert "NLR_DEVELOPER_API_KEY" in result.output
    # The error points at the keyless alternative.
    assert "--backend s3" in " ".join(result.output.split())


def test_confirmation_declined_aborts(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Declining the confirmation exits without fetching."""
    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    # Long temp paths truncate at the runner's 80 columns; render wide.
    from us_marine_energy_resource.cli._display import console as display_console

    monkeypatch.setattr(display_console, "_width", 220)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("declined confirm must not fetch")

    monkeypatch.setattr("us_marine_energy_resource.cli.wave.hindcast.get_data_at_point", boom)
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--backend", "api"], input="n\n")
    assert result.exit_code == 0
    # The confirmation names the server, the email, both destinations, and
    # the volume, which defaults to the most recent year.
    assert "developer.nlr.gov" in result.output
    assert "e@example.org" in result.output
    assert "significant_wave_height" in result.output
    assert "2,920" in result.output.replace("\n", "")  # one year of rows
    collapsed = " ".join(result.output.split())
    squeezed = collapsed.replace(" ", "")
    assert "Saves to" in collapsed and "point_44.5670_m124.2290_y2020-2020.csv" in squeezed
    assert "Cached at" in collapsed and tmp_path.name in squeezed
    assert "underthenamepoint_44.5670_m124.2290_y2020-2020" in squeezed


def test_fetch_with_yes_saves_csv_by_default(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--yes skips the prompt, prints the summary, and writes a CSV here."""
    import pandas as pd

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    frame = pd.DataFrame({"Significant Wave Height": [2.5, 3.0]})
    metadata = {
        "site": "mysite",
        "years": ["1979", "1980"],
        "variables": ["Significant Wave Height"],
    }
    calls: list[dict[str, Any]] = []

    def fake_get(lat: float, lon: float, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        calls.append(kwargs)
        return frame, metadata

    monkeypatch.setattr("us_marine_energy_resource.cli.wave.hindcast.get_data_at_point", fake_get)
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--yes", "--name", "mysite"])
    assert result.exit_code == 0
    assert "mysite" in result.output
    assert "2 rows" in result.output
    assert calls[0]["name"] == "mysite"
    assert calls[0]["return_metadata"] is True
    assert (cwd / "mysite.csv").exists()
    assert (cwd / "mysite_metadata.json").exists()


def test_defaults_narrow_to_latest_year_and_key_variables(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bare query fetches the four key variables for the most recent year."""
    import pandas as pd

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    monkeypatch.chdir(tmp_path)

    calls: list[dict[str, Any]] = []

    def fake_get(lat: float, lon: float, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        calls.append(kwargs)
        return pd.DataFrame({"x": [1.0]}), {"site": "s", "years": ["2020", "2020"], "variables": []}

    monkeypatch.setattr("us_marine_energy_resource.cli.wave.hindcast.get_data_at_point", fake_get)
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--yes", "--cache-only"])
    assert result.exit_code == 0
    assert calls[0]["years"] == [2020]
    assert calls[0]["variables"] == [
        "significant_wave_height",
        "energy_period",
        "peak_period",
        "omni-directional_wave_power",
    ]
    assert calls[0]["name"] == "point_44.5670_m124.2290_y2020-2020"
    # The user is told what was held back and how to get more.
    collapsed = " ".join(result.output.split())
    assert "only 2020 out of the served 1979-2020" in collapsed
    assert "--years" in collapsed and "--all" in collapsed
    # A small default query resolves to the keyless s3 backend.
    assert calls[0]["backend"] == "s3"


def test_all_fetches_everything_with_a_disclaimer(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--all sends no narrowing and the plan says it is heavy."""
    import pandas as pd

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    monkeypatch.chdir(tmp_path)

    calls: list[dict[str, Any]] = []

    def fake_get(lat: float, lon: float, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        calls.append(kwargs)
        return pd.DataFrame({"x": [1.0]}), {"site": "s", "years": ["1979", "2020"], "variables": []}

    monkeypatch.setattr("us_marine_energy_resource.cli.wave.hindcast.get_data_at_point", fake_get)
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--all", "--yes", "--cache-only"])
    assert result.exit_code == 0
    assert calls[0]["years"] is None and calls[0]["variables"] is None
    assert "everything the endpoint serves" in " ".join(result.output.split())
    # A large query with credentials configured resolves to the api backend.
    assert calls[0]["backend"] == "api"


def test_all_conflicts_with_narrowing(described: dict[str, Any]) -> None:
    """--all with --years is a usage error."""
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--all", "--years", "2020"])
    assert result.exit_code == 1
    assert "--all" in result.output


def test_summary_includes_statistics(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The result prints all-time and monthly statistics, skipping directions."""
    import numpy as np
    import pandas as pd

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    monkeypatch.chdir(tmp_path)
    # The statistics tables are wide; render like a real full-width terminal
    # instead of the runner's 80 columns, which would truncate the headers.
    from us_marine_energy_resource.cli._display import console as display_console

    monkeypatch.setattr(display_console, "_width", 220)

    index = pd.date_range("1979-01-01", periods=730, freq="12h", tz="UTC")
    frame = pd.DataFrame(
        {
            "Significant Wave Height": np.linspace(1.0, 5.0, len(index)),
            "Mean Wave Direction": np.full(len(index), 200.0),
        },
        index=index,
    )
    metadata = {
        "site": "mysite",
        "years": ["1979", "1979"],
        "variables": list(frame.columns),
        "units": {"Significant Wave Height": "m"},
    }
    monkeypatch.setattr(
        "us_marine_energy_resource.cli.wave.hindcast.get_data_at_point",
        lambda lat, lon, **kwargs: (frame, metadata),
    )
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--yes", "--cache-only"])
    assert result.exit_code == 0
    collapsed = " ".join(result.output.split())
    assert "Monthly Significant Wave Height [m] (1979 through 1979)" in collapsed
    for label in ("P0.1", "P1", "P5", "P25", "Median", "P75", "P95", "P99", "P99.9"):
        assert label in collapsed
    assert "Jan" in collapsed and "Dec" in collapsed
    # The all-time table comes after the monthly tables.
    assert collapsed.index("Monthly") < collapsed.index("All-time (")
    # Direction variables are angles, so they are excluded from the tables.
    assert "Mean Wave Direction" not in result.output
    # The save option is offered right where the tables are shown.
    assert "--stats-csv" in collapsed


def test_stats_csv_saves_the_tables(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--stats-csv writes one tidy CSV with every period and statistic."""
    import numpy as np
    import pandas as pd

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    monkeypatch.chdir(tmp_path)

    index = pd.date_range("1979-01-01", periods=730, freq="12h", tz="UTC")
    frame = pd.DataFrame(
        {"Significant Wave Height": np.linspace(1.0, 5.0, len(index))}, index=index
    )
    metadata = {"site": "mysite", "years": ["1979", "1979"], "variables": list(frame.columns)}
    monkeypatch.setattr(
        "us_marine_energy_resource.cli.wave.hindcast.get_data_at_point",
        lambda lat, lon, **kwargs: (frame, metadata),
    )
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--yes", "--cache-only", "--stats-csv"])
    assert result.exit_code == 0

    stats = pd.read_csv(tmp_path / "mysite_stats.csv")
    assert list(stats.columns) == [
        "Variable",
        "Period",
        "Min",
        "P0.1",
        "P1",
        "P5",
        "P25",
        "Median",
        "P75",
        "P95",
        "P99",
        "P99.9",
        "Max",
        "Mean",
    ]
    periods = stats[stats["Variable"] == "Significant Wave Height"]["Period"].tolist()
    assert periods[:3] == ["Jan", "Feb", "Mar"] and periods[-1] == "All-time"
    all_time = stats[stats["Period"] == "All-time"].iloc[0]
    assert all_time["Min"] == pytest.approx(1.0) and all_time["Max"] == pytest.approx(5.0)


def test_cache_only_writes_nothing_here(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--cache-only keeps the data in the cache and leaves the directory alone."""
    import pandas as pd

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    frame = pd.DataFrame({"Significant Wave Height": [2.5]})
    metadata = {"site": "mysite", "years": ["1979", "1979"], "variables": ["x"]}
    monkeypatch.setattr(
        "us_marine_energy_resource.cli.wave.hindcast.get_data_at_point",
        lambda lat, lon, **kwargs: (frame, metadata),
    )
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--yes", "--cache-only"])
    assert result.exit_code == 0
    assert list(cwd.iterdir()) == []


def test_output_dir_exports_csv(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """-o writes the combined CSV and metadata JSON."""
    import pandas as pd

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")

    frame = pd.DataFrame({"Significant Wave Height": [2.5]})
    metadata = {"site": "mysite", "years": ["1979", "1979"], "variables": ["x"]}
    monkeypatch.setattr(
        "us_marine_energy_resource.cli.wave.hindcast.get_data_at_point",
        lambda lat, lon, **kwargs: (frame, metadata),
    )
    out = tmp_path / "out"
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--yes", "-o", str(out)])
    assert result.exit_code == 0
    assert (out / "mysite.csv").exists()
    assert (out / "mysite_metadata.json").exists()


def _build_cache(root: Path) -> None:
    """Populate a wave cache: two sites, a pending request, archives, chunks."""
    import json

    for name, source in (("alpha", "api"), ("beta", "s3 direct")):
        site = root / f"{name}_44.57_-124.23"
        site.mkdir(parents=True)
        (site / f"{site.name}_2019-2020.csv").write_text("timestamp,x\n2019-01-01,1\n")
        (site / "metadata.json").write_text(
            json.dumps(
                {"site": name, "years": ["2019", "2020"], "variables": ["x"], "source": source}
            )
        )
    (root / "archives").mkdir()
    (root / "archives" / "alpha.zip").write_bytes(b"z" * 100)
    chunks = root / "s3_chunks" / "West_Coast" / "2020"
    chunks.mkdir(parents=True)
    (chunks / "significant_wave_height_0.npy").write_bytes(b"c" * 1000)
    (root / "requests.json").write_text(
        json.dumps({"alpha": {"download_url": "https://dl/a.zip"}, "ghost": {"download_url": None}})
    )


def test_cache_stats_lists_everything(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--cache shows sites, pending requests, chunk blocks, and the total."""
    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path))
    _build_cache(tmp_path)
    result = runner.invoke(app, ["wave", "--cache"])
    assert result.exit_code == 0
    collapsed = " ".join(result.output.split())
    assert "alpha" in collapsed and "beta" in collapsed
    assert "2019-2020" in collapsed
    assert "s3 direct" in collapsed
    assert "ghost" in collapsed  # requested but never downloaded
    assert "Chunk blocks" in collapsed
    assert "--clear NAME" in collapsed


def test_cache_clear_removes_one_item(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--clear removes the site dir, archive, and manifest entry, keeping the rest."""
    import json

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path))
    _build_cache(tmp_path)
    result = runner.invoke(app, ["wave", "--clear", "alpha", "-y"])
    assert result.exit_code == 0
    assert not (tmp_path / "alpha_44.57_-124.23").exists()
    assert not (tmp_path / "archives" / "alpha.zip").exists()
    assert (tmp_path / "beta_44.57_-124.23").exists()
    manifest = json.loads((tmp_path / "requests.json").read_text())
    assert "alpha" not in manifest and "ghost" in manifest


def test_cache_clear_pending_only_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--clear also removes a request that never produced a download."""
    import json

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path))
    _build_cache(tmp_path)
    result = runner.invoke(app, ["wave", "--clear", "ghost", "-y"])
    assert result.exit_code == 0
    assert "ghost" not in json.loads((tmp_path / "requests.json").read_text())


def test_cache_clear_unknown_name_lists_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An unknown name errors and names what the cache does hold."""
    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path))
    _build_cache(tmp_path)
    result = runner.invoke(app, ["wave", "--clear", "nope", "-y"])
    assert result.exit_code == 1
    assert "alpha" in result.output


def test_cache_clear_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--clear-all empties the cache after confirmation, and -y skips the prompt."""
    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path))
    _build_cache(tmp_path)
    declined = runner.invoke(app, ["wave", "--clear-all"], input="n\n")
    assert declined.exit_code == 0 and (tmp_path / "beta_44.57_-124.23").exists()

    result = runner.invoke(app, ["wave", "--clear-all", "-y"])
    assert result.exit_code == 0
    assert not tmp_path.exists()


def test_outage_warning_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    """A described point in a broken domain carries the outage warning."""
    broken = dict(_DESCRIBE, domain="Gulf_of_Mexico_and_Puerto_Rico")
    monkeypatch.setattr(
        "us_marine_energy_resource.cli.wave.hindcast.describe_point",
        lambda lat, lon, **kwargs: broken,
    )
    result = runner.invoke(app, ["wave", "18.6,-66.1", "--info"])
    assert result.exit_code == 0
    assert "not working" in result.output


def test_help_builders_are_total() -> None:
    """The prose helpers handle every input size without raising."""
    from us_marine_energy_resource.cli.wave import _count_word, _prose_list

    assert _prose_list([]) == ""
    assert _prose_list(["a"]) == "a"
    assert _prose_list(["a", "b"]) == "a and b"
    assert _prose_list(["a", "b", "c"]) == "a, b, and c"
    assert _count_word(4) == "four"
    assert _count_word(6) == "six"
    assert _count_word(99) == "99"
    assert _count_word(-1) == "-1"


def test_wave_help_derives_from_canonical_sources() -> None:
    """Every enumerable fact in the help comes from the defining modules."""
    from us_marine_energy_resource.cli.wave import _WAVE_EPILOG, _WAVE_HELP
    from us_marine_energy_resource.wave_hindcast.config import CONFIG
    from us_marine_energy_resource.wave_hindcast.domains import BASE_DOMAIN, DOMAINS

    for region in DOMAINS:
        assert region.replace("_", " ") in _WAVE_HELP
    assert f"{BASE_DOMAIN['first_year']}-{BASE_DOMAIN['last_year']}" in _WAVE_HELP
    assert CONFIG.api_key_env in _WAVE_HELP
    assert CONFIG.email_env in _WAVE_HELP
    assert CONFIG.s3_bucket_uri in _WAVE_HELP
    # The epilog's cache example names what a default fetch really creates.
    last = BASE_DOMAIN["last_year"]
    assert f"point_44.5700_m124.2300_y{last}-{last}" in _WAVE_EPILOG


def test_wave_help_builder_handles_domain_variants() -> None:
    """The builder stays total for edge-case domain tables."""
    from us_marine_energy_resource.cli.wave import _build_wave_help

    base = {"first_year": 1979, "last_year": 2020}
    uncapped = {"One": {}, "Two": {}}
    text = _build_wave_help(uncapped, base)
    assert "One and Two" in text and "only through" not in text

    capped = {"One": {}, "Two": {"last_year": 2010}}
    text = _build_wave_help(capped, base)
    assert "serves Two only through 2010" in text

    single = {"One": {}}
    assert "one region:" in _build_wave_help(single, base)


def test_wave_help_renders() -> None:
    """mer wave --help renders the interpolated help without error."""
    result = runner.invoke(app, ["wave", "--help"])
    assert result.exit_code == 0
    assert "WPTO" in result.output


def test_small_default_query_uses_s3_without_credentials(
    described: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bare small query reads S3 and never asks for a key."""
    import pandas as pd

    monkeypatch.setenv("MER_WAVE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("NLR_DEVELOPER_API_KEY", raising=False)
    monkeypatch.delenv("NLR_DEVELOPER_EMAIL", raising=False)
    # Keep the developer's own .env / ~/.mer.env out of the test.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: empty))

    calls: list[dict[str, Any]] = []

    def fake_get(lat: float, lon: float, **kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        calls.append(kwargs)
        return pd.DataFrame({"x": [1.0]}), {"site": "s", "years": ["2020", "2020"], "variables": []}

    monkeypatch.setattr("us_marine_energy_resource.cli.wave.hindcast.get_data_at_point", fake_get)
    result = runner.invoke(app, ["wave", PACWAVE_ARG, "--yes", "--cache-only"])
    assert result.exit_code == 0
    assert calls[0]["backend"] == "s3"
    assert "No API and no key" in " ".join(result.output.split())
