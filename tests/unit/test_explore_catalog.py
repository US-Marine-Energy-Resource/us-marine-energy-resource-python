"""Path resolution, capped listing, tree building, and the ls/info/download verbs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from us_marine_energy_resource.explore import catalog
from us_marine_energy_resource.explore.catalog import (
    FilePointer,
    LocalLister,
    PrefixPointer,
    S3Lister,
    build_tree,
    list_children,
    resolve_path,
)
from us_marine_energy_resource.mer import app

runner = CliRunner()


class FakeS3:
    """In-memory S3 that supports delimited (Delimiter='/') listing."""

    def __init__(self, sizes: dict[str, int]) -> None:
        """Store a mapping of object key -> size."""
        self.sizes = sizes

    def list_objects_v2(
        self,
        Bucket: str,  # noqa: N803 - boto3 argument names
        Prefix: str = "",  # noqa: N803
        Delimiter: str | None = None,  # noqa: N803
        MaxKeys: int = 1000,  # noqa: N803
        ContinuationToken: str | None = None,  # noqa: N803
    ) -> dict[str, Any]:
        """Return immediate children of Prefix, split into dirs and files."""
        dirs: set[str] = set()
        files: list[dict[str, Any]] = []
        for key in sorted(self.sizes):
            if not key.startswith(Prefix):
                continue
            rest = key[len(Prefix) :]
            if not rest:
                continue
            if "/" in rest:
                dirs.add(Prefix + rest.split("/", 1)[0] + "/")
            else:
                files.append({"Key": key, "Size": self.sizes[key]})
        return {
            "CommonPrefixes": [{"Prefix": d} for d in sorted(dirs)],
            "Contents": files,
            "IsTruncated": False,
        }


class PaginatedS3:
    """A fake that always reports more pages, to exercise the page bound."""

    def list_objects_v2(self, **kw: Any) -> dict[str, Any]:
        """Return one directory per call and always claim there is more."""
        i = int(kw.get("ContinuationToken") or "0")
        return {
            "CommonPrefixes": [{"Prefix": f"{kw['Prefix']}dir{i}/"}],
            "Contents": [],
            "IsTruncated": True,
            "NextContinuationToken": str(i + 1),
        }


_TIDAL = {
    "us-tidal/AK_cook_inlet/v1.0.0/b1/face_0.parquet": 100,
    "us-tidal/AK_cook_inlet/v1.0.0/b4_summary/s.parquet": 200,
    "us-tidal/WA_puget_sound/v1.0.0/x.parquet": 100,
    "us-tidal/manifest/m.json": 50,
}


def _lister() -> S3Lister:
    return S3Lister(FakeS3(_TIDAL), "b")


# --- path resolution -----------------------------------------------------------------------------


def test_resolve_endpoint_is_prefix() -> None:
    """Resolve endpoint is prefix."""
    p = resolve_path("tidal")
    assert (
        isinstance(p, PrefixPointer)
        and p.bucket == "marine-energy-data"
        and p.prefix == "us-tidal/"
    )


def test_resolve_endpoint_subpath_is_prefix() -> None:
    """Resolve endpoint subpath is prefix."""
    p = resolve_path("tidal/AK_cook_inlet")
    assert isinstance(p, PrefixPointer) and p.prefix == "us-tidal/AK_cook_inlet/"


def test_resolve_endpoint_file_by_extension() -> None:
    """A data extension in an endpoint sub-path resolves to a file."""
    p = resolve_path("tidal/AK_cook_inlet/v1.0.0/b4/summary.parquet")
    assert isinstance(p, FilePointer) and p.uri.startswith("s3://marine-energy-data/")


def test_resolve_s3_prefix_and_file() -> None:
    """Resolve s3 prefix and file."""
    assert isinstance(resolve_path("s3://bucket/some/prefix/"), PrefixPointer)
    assert isinstance(resolve_path("s3://bucket/file.h5"), FilePointer)


def test_resolve_local_dir_vs_file(tmp_path: Path) -> None:
    """A local directory is a prefix; a local path with no dir is a file."""
    (tmp_path / "sub").mkdir()
    assert isinstance(resolve_path(str(tmp_path)), PrefixPointer)
    assert isinstance(resolve_path(str(tmp_path / "x.h5")), FilePointer)


# --- listing -------------------------------------------------------------------------------------


def test_list_children_dirs_and_files() -> None:
    """List children dirs and files."""
    listing = list_children(_lister(), "us-tidal/")
    assert {e.name: e.is_dir for e in listing.entries} == {
        "AK_cook_inlet": True,
        "WA_puget_sound": True,
        "manifest": True,
    }
    assert not listing.truncated


def test_list_children_filter() -> None:
    """List children filter."""
    listing = list_children(_lister(), "us-tidal/", name_filter="AK_*")
    assert [e.name for e in listing.entries] == ["AK_cook_inlet"]


def test_list_children_truncates() -> None:
    """More children than the limit sets truncated."""
    listing = list_children(_lister(), "us-tidal/", limit=2)
    assert len(listing.entries) == 2 and listing.truncated


def test_list_children_bounds_pages() -> None:
    """A prefix that never stops paginating is bounded and marked truncated."""
    listing = list_children(S3Lister(PaginatedS3(), "b"), "huge/", limit=1000)
    assert listing.truncated and len(listing.entries) <= catalog._MAX_PAGES


def test_local_lister(tmp_path: Path) -> None:
    """Local lister lists a directory one level deep with sizes."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.h5").write_bytes(b"1234")
    listing = LocalLister().list_children(str(tmp_path) + "/", 100, None)
    names = {e.name: (e.is_dir, e.size) for e in listing.entries}
    assert names["sub"][0] is True
    assert names["a.h5"] == (False, 4)


def test_build_tree_depth() -> None:
    """Depth two expands each child directory one more level."""
    node = build_tree(_lister(), "us-tidal/", depth=2)
    ak = next(c for c in node.children if c.name == "AK_cook_inlet")
    assert ak.expanded and [c.name for c in ak.children] == ["v1.0.0"]


# --- CLI verbs -----------------------------------------------------------------------------------


def _patch_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route S3 listing through the in-memory fake."""
    monkeypatch.setattr(catalog, "make_client", lambda aws_profile=None: FakeS3(_TIDAL))


def test_ls_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ls endpoint."""
    _patch_client(monkeypatch)
    result = runner.invoke(app, ["ls", "tidal"])
    assert result.exit_code == 0 and "AK_cook_inlet/" in result.output


def test_ls_alias_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `list` alias works like `ls`."""
    _patch_client(monkeypatch)
    assert runner.invoke(app, ["list", "tidal"]).exit_code == 0


def test_info_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Info on a prefix shows a tree and an aggregate summary line."""
    _patch_client(monkeypatch)
    result = runner.invoke(app, ["info", "tidal"])
    assert result.exit_code == 0 and "shown" in result.output


def test_info_alias_i(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `i` alias works like `info`."""
    _patch_client(monkeypatch)
    assert runner.invoke(app, ["i", "tidal"]).exit_code == 0


def test_explore_browses_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explore browses prefix."""
    _patch_client(monkeypatch)
    result = runner.invoke(app, ["explore", "tidal"])
    assert result.exit_code == 0 and "AK_cook_inlet" in result.output


def test_explore_mode_flag_on_prefix_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mode flag on a directory is rejected."""
    _patch_client(monkeypatch)
    result = runner.invoke(app, ["explore", "tidal", "--head"])
    assert result.exit_code == 1 and "apply to files" in result.output


def test_info_file_shows_global_and_variable_attrs(h5_file: Path) -> None:
    """Info on an h5 file shows the root attributes and each variable's attrs."""
    result = runner.invoke(app, ["info", str(h5_file)])
    assert result.exit_code == 0
    assert "format" in result.output and "root attributes" in result.output
    assert "title" in result.output  # a root attribute
    assert "significant_wave_height" in result.output  # a variable
    assert "scale_factor" in result.output  # that variable's attribute


def test_info_parquet_shows_schema_and_file_metadata(tmp_path: Path) -> None:
    """Info on a parquet file shows file-level metadata and per-column metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    speed = pa.field("speed", pa.float32(), metadata={"units": "m s-1"})
    schema = pa.schema([speed, pa.field("direction", pa.int32())]).with_metadata(
        {"source": "unit-test", "location": "Cook Inlet"}
    )
    table = pa.table(
        {"speed": pa.array([1.0, 2.0], pa.float32()), "direction": pa.array([10, 20], pa.int32())},
        schema=schema,
    )
    path = tmp_path / "meta.parquet"
    pq.write_table(table, path)

    result = runner.invoke(app, ["info", str(path)])
    assert result.exit_code == 0
    assert "source" in result.output and "unit-test" in result.output  # file metadata
    assert "/speed" in result.output  # a column
    assert "units" in result.output and "m s-1" in result.output  # its schema metadata
    assert "/direction" in result.output  # a column with no metadata still lists


def _serve_http(monkeypatch: pytest.MonkeyPatch, blob: bytes) -> None:
    """Serve a blob as a range-capable fake requests module."""
    import importlib
    import types

    class Response:
        def __init__(self, content: bytes, headers: dict[str, str]) -> None:
            self.content = content
            self.headers = headers
            self.status_code = 206

        def raise_for_status(self) -> None:
            pass

    def head(url: str, allow_redirects: bool = True, timeout: int = 30) -> Response:
        return Response(b"", {"Accept-Ranges": "bytes", "Content-Length": str(len(blob))})

    def get(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> Response:
        start, end = (headers or {})["Range"].removeprefix("bytes=").split("-")
        return Response(blob[int(start) : int(end) + 1], {})

    real_import = importlib.import_module
    mod = types.SimpleNamespace(head=head, get=get)
    monkeypatch.setattr(
        "us_marine_energy_resource.explore.lazy.importlib.import_module",
        lambda name: mod if name == "requests" else real_import(name),
    )


def test_info_remote_h5_shows_header_first(h5_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote HDF5 file shows the quick header and offers --variables."""
    _serve_http(monkeypatch, h5_file.read_bytes())
    result = runner.invoke(app, ["info", "https://example.org/data.h5"])
    assert result.exit_code == 0
    assert "root attributes" in result.output and "title" in result.output
    assert "--variables" in result.output  # the drill-down hint
    assert "scale_factor" not in result.output  # variable attrs not walked


def test_info_remote_h5_variables_flag_walks(
    h5_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--variables collects each variable's attributes from a remote file."""
    _serve_http(monkeypatch, h5_file.read_bytes())
    result = runner.invoke(app, ["info", "https://example.org/data.h5", "--variables"])
    assert result.exit_code == 0
    assert "significant_wave_height" in result.output
    assert "scale_factor" in result.output


def test_info_falls_back_to_header_when_budget_trips(
    h5_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file whose metadata blows the walk budget still shows its header."""
    from us_marine_energy_resource.explore.errors import TransferBudgetExceededError
    from us_marine_energy_resource.explore.formats.hdf5 import Hdf5OpenFile

    def blow(self: Hdf5OpenFile, *, storage: bool = False) -> None:
        raise TransferBudgetExceededError(fetched=150 * 1024 * 1024, limit=100 * 1024 * 1024)

    monkeypatch.setattr(Hdf5OpenFile, "summary", blow)
    result = runner.invoke(app, ["info", str(h5_file)])
    assert result.exit_code == 0
    assert "root attributes" in result.output  # the cheap header still renders
    assert "stopped" in result.output and "--max-transfer-mb" in result.output


_WAVE = {
    "v1.0.1/West_Coast/West_Coast_wave_2010.h5": 80_000_000_000,
    "v1.0.1/West_Coast/West_Coast_wave_2011.h5": 81_000_000_000,
}


def test_download_prefix_with_no_files_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prefix holding only subdirectories has nothing to download."""
    _patch_client(monkeypatch)
    result = runner.invoke(app, ["download", "tidal"])
    assert result.exit_code == 1 and "no files" in result.output


def test_download_wave_domain_refused_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """A multi-TB wave domain is refused, pointing at mer wave for point data."""
    monkeypatch.setattr(catalog, "make_client", lambda aws_profile=None: FakeS3(_WAVE))
    result = runner.invoke(app, ["download", "wave/v1.0.1/West_Coast"])
    assert result.exit_code == 1
    assert "--max-download-mb" in result.output
    assert "mer wave" in result.output


def test_download_prefix_local_dir(tmp_path: Path) -> None:
    """One level of a local directory downloads; subdirectories are skipped."""
    src = tmp_path / "src"
    (src / "deeper").mkdir(parents=True)
    (src / "a.parquet").write_bytes(b"PAR1aaaa")
    (src / "b.parquet").write_bytes(b"PAR1bbbb")
    (src / "deeper" / "c.parquet").write_bytes(b"PAR1cccc")
    out = tmp_path / "out"

    result = runner.invoke(app, ["download", str(src), "-o", str(out)])
    assert result.exit_code == 0
    assert (out / "a.parquet").read_bytes() == b"PAR1aaaa"
    assert (out / "b.parquet").read_bytes() == b"PAR1bbbb"
    assert not (out / "c.parquet").exists()
    assert "skipping 1" in result.output

    # A second run finds the files already present and moves nothing.
    again = runner.invoke(app, ["download", str(src), "-o", str(out)])
    assert again.exit_code == 0
    assert "exists" in again.output


def test_download_local_file(h5_file: Path, tmp_path: Path) -> None:
    """Download of a local file copies it to the output directory."""
    out = tmp_path / "out"
    result = runner.invoke(app, ["download", str(h5_file), "-o", str(out)])
    assert result.exit_code == 0
    assert (out / h5_file.name).exists()
    assert (out / h5_file.name).read_bytes() == h5_file.read_bytes()


def test_download_alias_dl(h5_file: Path, tmp_path: Path) -> None:
    """The `dl` alias works like `download`."""
    result = runner.invoke(app, ["dl", str(h5_file), "-o", str(tmp_path / "o")])
    assert result.exit_code == 0
