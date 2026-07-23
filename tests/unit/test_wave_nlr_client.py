"""The NLR download API client: classification, credentials, and requests."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from tests.unit.conftest import FakeResponse, FakeTime, _patch_requests
from us_marine_energy_resource.wave_hindcast import errors
from us_marine_energy_resource.wave_hindcast.config import CONFIG
from us_marine_energy_resource.wave_hindcast.nlr_api import client
from us_marine_energy_resource.wave_hindcast.nodes import WaveNode

NODE = WaveNode(
    location_id=479519,
    domain="West_Coast",
    endpoint="us-west-coast-hindcast-download",
    lat=44.5682,
    lon=-124.228,
    distance_m=142.3,
)


# --------------------------------------------------------------------------- #
# classify
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (429, {"errors": ["slow down"]}, errors.RateLimitError),
        (403, {"errors": ["bad key"]}, errors.AuthenticationError),
        (404, {"errors": ["nope"]}, errors.EndpointNotFoundError),
        (
            400,
            {"errors": ["Values may include significant_wave_height, energy_period"]},
            errors.InvalidAttributeError,
        ),
        (400, {"errors": ["the queue is full right now"]}, errors.QueueFullError),
        (
            400,
            {"errors": ["No data available at the provided location"]},
            errors.NoDataAtLocationError,
        ),
        (400, {"errors": ["Invalid value(s) for names: 2015"]}, errors.InvalidYearError),
        (500, {"errors": ["mystery"]}, errors.RequestError),
    ],
)
def test_classify(status: int, body: dict[str, Any], expected: type) -> None:
    """Each observed failure phrase maps to its own exception class."""
    exc = client.classify(FakeResponse(status, body))
    assert type(exc) is expected
    assert exc.status == status


def test_classify_extracts_valid_attributes() -> None:
    """An attribute rejection carries the endpoint's own list."""
    exc = client.classify(
        FakeResponse(400, {"errors": ["Values may include significant_wave_height, energy_period"]})
    )
    assert isinstance(exc, errors.InvalidAttributeError)
    assert exc.valid == ["significant_wave_height", "energy_period"]


def test_classify_non_json_body() -> None:
    """A non-JSON body still classifies, keeping a text excerpt."""
    exc = client.classify(FakeResponse(500, None, text="<html>boom</html>"))
    assert isinstance(exc, errors.RequestError)
    assert "boom" in " ".join(exc.errors)


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #


def test_credentials_missing_names_both(_no_cred_env: Path) -> None:
    """The error names exactly the unset variables."""
    with pytest.raises(errors.CredentialsMissingError) as excinfo:
        client.credentials()
    assert "NLR_DEVELOPER_API_KEY" in str(excinfo.value)
    assert "NLR_DEVELOPER_EMAIL" in str(excinfo.value)


def test_credentials_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both set means both returned."""
    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "k")
    monkeypatch.setenv("NLR_DEVELOPER_EMAIL", "e@example.org")
    assert client.credentials() == ("k", "e@example.org")


@pytest.fixture
def _no_cred_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Clear the credential env vars and isolate cwd and home."""
    monkeypatch.delenv("NLR_DEVELOPER_API_KEY", raising=False)
    monkeypatch.delenv("NLR_DEVELOPER_EMAIL", raising=False)
    cwd = tmp_path / "project"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    return tmp_path


def test_credentials_from_local_env_file(_no_cred_env: Path) -> None:
    """A .env in the current directory supplies both values."""
    (Path.cwd() / ".env").write_text(
        "# wave credentials\n"
        "export NLR_DEVELOPER_API_KEY='local-key'\n"
        'NLR_DEVELOPER_EMAIL="local@example.org"\n'
    )
    assert client.credentials() == ("local-key", "local@example.org")


def test_credentials_from_global_env_file(_no_cred_env: Path) -> None:
    """~/.mer.env supplies values from any working directory."""
    (Path.home() / ".mer.env").write_text(
        "NLR_DEVELOPER_API_KEY=global-key\nNLR_DEVELOPER_EMAIL=global@example.org\n"
    )
    assert client.credentials() == ("global-key", "global@example.org")


def test_credentials_precedence(_no_cred_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment beats the local .env, which beats ~/.mer.env, per value."""
    (Path.home() / ".mer.env").write_text(
        "NLR_DEVELOPER_API_KEY=global-key\nNLR_DEVELOPER_EMAIL=global@example.org\n"
    )
    (Path.cwd() / ".env").write_text("NLR_DEVELOPER_API_KEY=local-key\n")
    assert client.credentials() == ("local-key", "global@example.org")

    monkeypatch.setenv("NLR_DEVELOPER_API_KEY", "env-key")
    assert client.credentials() == ("env-key", "global@example.org")


def test_credentials_missing_everywhere_names_the_places(_no_cred_env: Path) -> None:
    """The error explains all three places a value can live."""
    with pytest.raises(errors.CredentialsMissingError) as excinfo:
        client.credentials()
    message = str(excinfo.value)
    assert ".env" in message and ".mer.env" in message


# --------------------------------------------------------------------------- #
# post
# --------------------------------------------------------------------------- #


def test_post_key_in_params_payload_in_body(
    monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime
) -> None:
    """The API key travels as a query parameter, everything else in the body."""
    seen: dict[str, Any] = {}

    def fake_post(url: str, params: dict[str, str], data: dict[str, str], timeout: int) -> Any:
        seen.update(url=url, params=params, data=data)
        return FakeResponse(200, {"outputs": {}})

    _patch_requests(monkeypatch, types.SimpleNamespace(post=fake_post))
    response = client.post("some-endpoint", "KEY", {"names": "1979"})
    assert response.status_code == 200
    assert seen["url"] == f"{CONFIG.api_base_url}/some-endpoint.json"
    assert seen["params"] == {"api_key": "KEY"}
    assert seen["data"] == {"names": "1979"}


def test_post_retries_past_429(monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime) -> None:
    """429s back off and retry rather than losing the request."""
    responses = [FakeResponse(429, {"errors": ["slow"]}), FakeResponse(200, {"outputs": {}})]
    events: list[str] = []

    def fake_post(url: str, params: dict[str, str], data: dict[str, str], timeout: int) -> Any:
        return responses.pop(0)

    _patch_requests(monkeypatch, types.SimpleNamespace(post=fake_post))
    response = client.post("ep", "KEY", {}, events.append)
    assert response.status_code == 200
    assert any("rate limited" in e for e in events)
    assert client.RATE_LIMIT_BACKOFF_S in fake_time.slept


def test_post_enforces_spacing(monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime) -> None:
    """Two back-to-back posts sleep out the minimum request spacing."""
    _patch_requests(
        monkeypatch,
        types.SimpleNamespace(post=lambda url, **kwargs: FakeResponse(200, {"outputs": {}})),
    )
    monkeypatch.setattr(client, "_last_post", [0.0])
    client.post("ep", "KEY", {})
    client.post("ep", "KEY", {})
    assert any(0 < s <= client.REQUEST_DELAY_S for s in fake_time.slept)


# --------------------------------------------------------------------------- #
# request_node
# --------------------------------------------------------------------------- #


def test_request_node_payload(monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime) -> None:
    """The request uses location_ids, all years, and states utc/leap_day."""
    monkeypatch.setitem(client._attribute_cache, "West_Coast", ["significant_wave_height"])
    seen: dict[str, Any] = {}

    def fake_post(url: str, params: dict[str, str], data: dict[str, str], timeout: int) -> Any:
        seen.update(data=data)
        return FakeResponse(
            200, {"outputs": {"downloadUrl": "https://dl/abc.zip", "message": "queued"}}
        )

    _patch_requests(monkeypatch, types.SimpleNamespace(post=fake_post))
    entry = client.request_node(NODE, 44.567, -124.229, "KEY", "e@example.org")

    data = seen["data"]
    assert data["location_ids"] == "479519"
    assert data["names"].startswith("1979,") and data["names"].endswith(",2020")
    assert data["utc"] == "true"
    assert data["leap_day"] == "true"
    assert data["email"] == "e@example.org"
    assert entry["download_url"] == "https://dl/abc.zip"
    assert entry["located_by"] == "location_ids"
    assert entry["years"] == ["1979", "2020"]


def test_request_node_reports_quota(monkeypatch: pytest.MonkeyPatch, fake_time: FakeTime) -> None:
    """Rate-limit headers surface as a quota event and land in the manifest."""
    monkeypatch.setitem(client._attribute_cache, "West_Coast", ["significant_wave_height"])
    events: list[str] = []

    def fake_post(url: str, params: dict[str, str], data: dict[str, str], timeout: int) -> Any:
        return FakeResponse(
            200,
            {"outputs": {"downloadUrl": "https://dl/a.zip"}},
            headers={"X-RateLimit-Remaining": "1993", "X-RateLimit-Limit": "2000"},
        )

    _patch_requests(monkeypatch, types.SimpleNamespace(post=fake_post))
    entry = client.request_node(NODE, 44.567, -124.229, "KEY", "e@example.org", events.append)
    assert any(e == "quota: 1993 of 2000 daily API requests remaining" for e in events)
    assert entry["rate_limit_remaining"] == "1993"


def test_response_json_rejects_200_with_errors() -> None:
    """A 200 carrying a populated errors array is still a rejection."""
    with pytest.raises(errors.RequestError):
        client._response_json(FakeResponse(200, {"errors": ["Invalid value(s)"], "outputs": {}}))
