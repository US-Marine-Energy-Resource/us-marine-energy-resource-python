"""Talk to the NLR developer download API.

Credentials, the paced POST, error classification, the attribute probe, and
the request that queues a grid node's record. The API returns a node's whole
record in a single request, and downloading and organizing the resulting
archive is :mod:`.archive`'s job.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...explore.lazy import lazy_import
from .. import errors
from .._store import _noop
from ..config import CONFIG
from ..domains import domain_config, years_for
from ..nodes import WaveNode

# Attributes are discovered per domain rather than hard-coded, because the
# docs list attributes the endpoints reject and spellings vary by domain.
# Rejecting a bad attribute makes the API enumerate the ones it does accept,
# and the 400 is refused before any job is queued, so probing costs nothing.
PROBE_ATTRIBUTE = "__probe__"
VALID_ATTRS_RE = re.compile(r"Values may include (?P<attrs>[a-z0-9_,\- ]+)", re.I)

# The documented request-size cap is far above a whole single-node record, so
# one request per site always fits.

# Non-CSV endpoints allow one request every two seconds, and every POST counts,
# including the attribute probe. Pacing lives in `post` so no call site forgets.
REQUEST_DELAY_S = 2

# 429s still happen despite the pacing, because the daily and in-flight limits
# are separate from the per-second one. Retry rather than lose a site.
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BACKOFF_S = 30


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from an env file, empty when it does not exist.

    Understands comments, blank lines, an optional ``export`` prefix, and
    single or double quotes around the value.

    Parameters
    ----------
    path : Path
        The env file to read.

    Returns
    -------
    dict
        Parsed entries, empty when the file is missing or unreadable.
    """
    entries: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return entries
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        entries[key.strip()] = value.strip().strip("'\"")
    return entries


def _credential(name: str) -> str | None:
    """Look one credential up: environment, then ./.env, then ~/.mer.env.

    Parameters
    ----------
    name : str
        Environment variable name.

    Returns
    -------
    str or None
        The value, or None when unset everywhere.
    """
    value = os.environ.get(name)
    if value:
        return value
    for path in (Path.cwd() / ".env", Path.home() / ".mer.env"):
        value = _read_env_file(path).get(name)
        if value:
            return value
    return None


def credentials() -> tuple[str, str]:
    """Read the API key and contact email.

    Each value is looked up in the environment, then a ``.env`` file in the
    current directory, then ``~/.mer.env``.

    Returns
    -------
    tuple of (str, str)
        ``(api_key, email)``.

    Raises
    ------
    CredentialsMissingError
        Either value is unset everywhere. The message names the missing ones.
    """
    api_key = _credential(CONFIG.api_key_env)
    email = _credential(CONFIG.email_env)
    missing = [
        name
        for name, value in (
            (CONFIG.api_key_env, api_key),
            (CONFIG.email_env, email),
        )
        if not value
    ]
    if missing:
        raise errors.CredentialsMissingError(
            f"not set: {', '.join(missing)}. Set them in the environment, in a "
            ".env file in the current directory, or in ~/.mer.env. A free key "
            f"is available at {CONFIG.signup_url}"
        )
    assert api_key is not None and email is not None
    return api_key, email


def _parse_attrs(match: re.Match[str]) -> list[str]:
    """Split a matched attribute enumeration into clean names.

    Parameters
    ----------
    match : re.Match
        A :data:`VALID_ATTRS_RE` match.

    Returns
    -------
    list of str
        The attribute names the endpoint enumerated.
    """
    return [a.strip() for a in match.group("attrs").split(",") if a.strip()]


def classify(response: Any) -> errors.RequestError:
    """Turn a rejected response into the most specific error available.

    The service reports the cause in prose, so this matches on the phrases it
    actually emits and falls back to the base RequestError.

    Parameters
    ----------
    response : requests.Response
        The rejected response.

    Returns
    -------
    RequestError
        Ready to raise; never raised here.
    """
    status = response.status_code
    try:
        body = response.json()
        reported = body.get("errors") or []
        if isinstance(body.get("error"), dict):
            reported = [body["error"].get("message", "")]
    except ValueError:
        reported = [response.text[:200]]
    joined = " ".join(reported)

    if status == 429:
        return errors.RateLimitError("rate limit exceeded", status, reported)
    if status == 403:
        return errors.AuthenticationError("API key rejected", status, reported)
    if status == 404:
        return errors.EndpointNotFoundError("no such endpoint", status, reported)

    match = VALID_ATTRS_RE.search(joined)
    if match:
        return errors.InvalidAttributeError(
            "attribute not available for this domain", status, reported, _parse_attrs(match)
        )
    if "queue" in joined.lower():
        return errors.QueueFullError("service queue is full", status, reported)
    if "No data available at the provided location" in joined:
        return errors.NoDataAtLocationError("no grid node at that location", status, reported)
    if "Invalid value" in joined:
        return errors.InvalidYearError("year outside this domain's range", status, reported)
    return errors.RequestError(f"request rejected ({status})", status, reported)


_last_post = [0.0]


def post(
    endpoint: str,
    api_key: str,
    payload: dict[str, str],
    on_event: Callable[[str], None] = _noop,
) -> Any:
    """POST to an endpoint slug, respecting the rate limit and retrying past 429s.

    Every request funnels through here so the minimum spacing is enforced
    once. The attribute probe counts as a real request too.

    Parameters
    ----------
    endpoint : str
        Endpoint slug without the ``.json`` suffix.
    api_key : str
        NLR developer API key.
    payload : dict
        Form fields for the POST body.
    on_event : callable
        Sink for progress messages.

    Returns
    -------
    requests.Response
        The last response received, which may still be a 429.
    """
    requests = lazy_import("requests", "calling the NLR wave download API")
    response = None
    for attempt in range(RATE_LIMIT_RETRIES):
        wait = REQUEST_DELAY_S - (time.monotonic() - _last_post[0])
        if wait > 0:
            time.sleep(wait)
        try:
            response = requests.post(
                f"{CONFIG.api_base_url}/{endpoint}.json",
                # The docs are explicit that on POST the key must be a query
                # parameter even though everything else travels in the body.
                params={"api_key": api_key},
                data=payload,
                timeout=60,
            )
        except requests.RequestException as exc:
            # The exception text embeds the request URL, and the URL carries
            # the API key. Redact it, and drop the chained traceback so no
            # code path can print the raw URL.
            detail = str(exc).replace(api_key, "***")
            raise errors.RequestError(f"could not reach the API: {detail}") from None
        _last_post[0] = time.monotonic()
        if response.status_code != 429:
            return response
        if attempt < RATE_LIMIT_RETRIES - 1:
            delay = RATE_LIMIT_BACKOFF_S * (attempt + 1)
            on_event(f"rate limited, retrying in {delay}s")
            time.sleep(delay)
    return response


def _response_json(response: Any) -> dict[str, Any]:
    """Parse the response body, raising the classified error on any failure.

    The API answers 200 with a populated ``errors`` list rather than an HTTP
    error code for some failures, so both have to be checked.

    Parameters
    ----------
    response : requests.Response
        The response to parse.

    Returns
    -------
    dict
        The parsed body.

    Raises
    ------
    RequestError
        The status was not 200, the body was not JSON, or the body carried
        a populated ``errors`` array.
    """
    if response.status_code != 200:
        raise classify(response)
    try:
        payload = response.json()
    except ValueError:
        raise errors.RequestError("response was not JSON", response.status_code) from None
    # A populated `errors` array with a 200 status is a rejection too.
    if payload.get("errors"):
        raise classify(response)
    return payload


_attribute_cache: dict[str, list[str]] = {}


def attributes_for(
    domain: str,
    api_key: str,
    email: str,
    on_event: Callable[[str], None] = _noop,
) -> list[str]:
    """Ask the API which attributes a domain's endpoint actually accepts.

    Deliberately does not fall back to a hard-coded list on failure: if we
    cannot establish what is available, silently downloading some other column
    set is worse than stopping.

    Parameters
    ----------
    domain : str
        Domain name.
    api_key, email : str
        Credentials.
    on_event : callable
        Sink for progress messages.

    Returns
    -------
    list of str
        The attributes the endpoint enumerated as acceptable.
    """
    if domain in _attribute_cache:
        return _attribute_cache[domain]

    config = domain_config(domain)
    response = post(
        config["endpoint"],
        api_key,
        {
            # A point inside the domain, not a dummy: GOM/PR accepts an
            # out-of-domain point with a 200 and only fails later, mailing the
            # user a download-error notice.
            "wkt": config["probe_point"],
            "names": str(config["first_year"]),
            "interval": config["interval"],
            "email": email,
            "attributes": PROBE_ATTRIBUTE,
        },
        on_event,
    )
    try:
        reported = " ".join(response.json().get("errors") or [])
    except ValueError:
        reported = response.text
    match = VALID_ATTRS_RE.search(reported)
    if not match:
        raise classify(response)
    attributes = _parse_attrs(match)
    _attribute_cache[domain] = attributes
    return attributes


def request_node(
    node: WaveNode,
    requested_lat: float,
    requested_lon: float,
    api_key: str,
    email: str,
    on_event: Callable[[str], None] = _noop,
    years_subset: list[int] | None = None,
    variables: list[str] | None = None,
) -> dict[str, Any]:
    """Queue the record for one grid node. Returns its manifest entry.

    Parameters
    ----------
    node : WaveNode
        The resolved grid node.
    requested_lat, requested_lon : float
        The coordinate the caller originally asked for, kept for metadata.
    api_key, email : str
        Credentials.
    on_event : callable
        Sink for progress messages.
    years_subset, variables : list, optional
        Narrow the request; ``None`` means everything the domain serves.

    Returns
    -------
    dict
        The manifest entry, including ``download_url``.
    """
    config = domain_config(node.domain)
    years = years_for(config)
    if years_subset:
        years = [y for y in years if int(y) in set(years_subset)]
        if not years:
            raise errors.InvalidYearError(
                f"this domain serves {config['first_year']}-{config['last_year']}"
            )
    attributes = attributes_for(node.domain, api_key, email, on_event)
    if variables:
        missing = sorted(set(variables) - set(attributes))
        if missing:
            raise errors.InvalidAttributeError(
                f"not served by this endpoint: {', '.join(missing)}",
                valid=attributes,
            )
        attributes = [a for a in attributes if a in set(variables)]
    on_event(f"{len(attributes)} attributes available: {', '.join(attributes)}")
    on_event(f"location_id {node.location_id} ({node.distance_m:.0f} m from the requested point)")

    payload = {
        # `location_ids` is undocumented but is what the Marine Energy Atlas
        # itself emits. The documented `wkt` path fails server-side for many
        # valid points that build fine when requested by id.
        "location_ids": str(node.location_id),
        "names": ",".join(years),
        "interval": config["interval"],
        "attributes": ",".join(attributes),
        # Stated rather than left to the defaults: leap day defaults to off,
        # which silently drops Feb 29 and leaves leap years short of the .h5
        # files published on S3.
        "utc": config["utc"],
        config["leap_day_param"]: "true",
        "email": email,
    }
    response = post(config["endpoint"], api_key, payload, on_event)
    outputs = _response_json(response)["outputs"]
    # The api.data.gov gateway reports the daily quota on every response.
    headers = getattr(response, "headers", None) or {}
    remaining = headers.get("X-RateLimit-Remaining")
    limit = headers.get("X-RateLimit-Limit")
    if remaining is not None and limit is not None:
        on_event(f"quota: {remaining} of {limit} daily API requests remaining")
    return {
        "domain": node.domain,
        "requested_lat": requested_lat,
        "requested_lon": requested_lon,
        "location_id": node.location_id,
        "located_by": "location_ids",
        "node_distance_m": round(node.distance_m, 1),
        "years": [years[0], years[-1]],
        "interval_minutes": int(config["interval"]),
        "attributes": attributes,
        "leap_day_param": config["leap_day_param"],
        "utc": True,
        "leap_day": True,
        "download_url": outputs.get("downloadUrl"),
        "message": outputs.get("message"),
        "rate_limit_remaining": remaining,
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
