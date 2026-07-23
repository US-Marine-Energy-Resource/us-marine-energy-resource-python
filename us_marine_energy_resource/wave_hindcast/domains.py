"""Facts about the WPTO wave hindcast domains.

Everything here was established against the live service or the published S3
bucket, not the documentation, and the comments record where the two disagree.
Backends and the node index both read from this module, so a backend can
differ from the download API where the underlying data really differs.
Stdlib-only at import, safe to import from anywhere in the package.
"""

from __future__ import annotations

from typing import Any

from . import errors
from .config import CONFIG

# Endpoint settings shared by every domain, overridden per domain below.
#
# `leap_day_param` varies across endpoints, and a wrong name is accepted with
# a 200 and silently ignored, costing Feb 29 with no error to notice. Verify
# any override by counting rows in a leap year, never assume it.
BASE_DOMAIN: dict[str, Any] = {
    "interval": "180",
    "first_year": 1979,
    "last_year": 2020,
    "leap_day_param": "leap_day",
    "utc": "true",
    # Correction applied to direction columns to bring them into the
    # meteorological convention (degrees clockwise from north, direction the
    # waves come FROM). None means the domain already uses it.
    "direction_transform": None,
}

# Converts atan2 output (degrees counter-clockwise from east, direction of
# travel) into the meteorological convention.
MATH_TO_MET = "270-x"

# Per-domain overrides. `endpoint` is the download-API slug, `probe_point` is a
# point inside the domain for the attribute probe, and `grid_key` is the one S3
# file the node grid can be read from (2010, later files dropped `coordinates`).
DOMAINS: dict[str, dict[str, Any]] = {
    "West_Coast": {
        "endpoint": "us-west-coast-hindcast-download",
        "probe_point": "POINT (-124.2 46.2)",
        "grid_key": "v1.0.1/West_Coast/West_Coast_wave_2010.h5",
        # Verified: leap_day=true yields 2928-row leap years.
    },
    "Atlantic": {
        "endpoint": "us-atlantic-hindcast-download",
        "probe_point": "POINT (-80.643 24.382)",
        "grid_key": "v1.0.1/Atlantic/Atlantic_wave_2010.h5",
        # The docs' prose claims 1979-2020, but the endpoint rejects 2011 and
        # later: 32 of the 42 years. S3 publishes all 42 Atlantic years, so
        # the cap is the API's alone.
        "last_year": 2010,
    },
    "Hawaii": {
        "endpoint": "hawaii-hindcast-download",
        "probe_point": "POINT (-164 15)",
        "grid_key": "v1.0.0/Hawaii/Hawaii_wave_2010.h5",
        # Same prose/table contradiction as Atlantic, same verified cutoff.
        "last_year": 2010,
        # Hawaii ships its directions in the atan2 convention while every
        # other domain uses meteorological. Left uncorrected, the wave energy
        # at Oahu's windward WETS node arrives straight through the island to
        # its west. Checked against the same node's mean_wave_direction on S3:
        # 270-x gives about 5 deg median error over the most energetic records,
        # versus about 128 deg for the raw values. A leeward Big Island control
        # node, whose waves must arrive from the west, agrees.
        "direction_transform": MATH_TO_MET,
    },
    "Alaska": {
        "endpoint": "alaska-hindcast-download",
        "probe_point": "POINT (-160.491356 54.040203)",
        "grid_key": "v1.0.1/Alaska/Alaska_wave_2010.h5",
    },
    "Gulf_of_Mexico_and_Puerto_Rico": {
        "endpoint": "us-wave-v1-0-0-gom-and-pr-download",
        "probe_point": "POINT (-67.97 17.62)",
        "grid_key": "v1.0.1/Gulf_of_Mexico_and_Puerto_Rico/GOM_PR_2010.h5",
    },
    "CNMI_and_Guam": {
        "endpoint": "us-wave-v1-0-0-cnmi-and-guam-download",
        "probe_point": "POINT (145.6704112711147 16.32973014436503)",
        "grid_key": "v1.0.0/CNMI_and_Guam/CNMI_and_Guam_wave_2010.h5",
    },
}

# Download endpoint per domain, for callers that only need the slug.
DOMAIN_ENDPOINTS: dict[str, str] = {name: cfg["endpoint"] for name, cfg in DOMAINS.items()}


def domain_config(domain: str) -> dict[str, Any]:
    """Merge the base settings with this domain's overrides.

    Parameters
    ----------
    domain : str
        One of the keys of :data:`DOMAINS`.

    Returns
    -------
    dict
        The merged configuration.
    """
    config = dict(BASE_DOMAIN)
    config.update(DOMAINS[domain])
    return config


def years_for(config: dict[str, Any]) -> list[str]:
    """Every year a domain serves, as the strings the API wants.

    Parameters
    ----------
    config : dict
        A :func:`domain_config` result.

    Returns
    -------
    list of str
        ``first_year`` through ``last_year`` inclusive.
    """
    return [str(y) for y in range(config["first_year"], config["last_year"] + 1)]


# Domains whose API download endpoint accepts requests it never fulfils. The
# outage is the API's alone: the same data still reads fine from S3, so only
# the api backend fails fast on these.
#
# Recheck by removing the entry and running:
#     mer wave 18.6,-66.1 --force --yes
API_OUTAGES: dict[str, str] = {
    "Gulf_of_Mexico_and_Puerto_Rico": (
        "Every request to this endpoint failed on 2026-07-20/21, with both "
        "`wkt` and `location_ids` at multiple grid nodes. The service returns "
        "200 with a download URL, then the archive never appears and a "
        f"failure email arrives instead. Report to {CONFIG.support_email}."
    ),
}


def check_api_outage(domain: str) -> None:
    """Fail fast on a domain whose API accepts requests it never fulfils.

    Parameters
    ----------
    domain : str
        Domain name to check.

    Raises
    ------
    ApiOutageError
        The domain is listed in :data:`API_OUTAGES`.
    """
    if domain in API_OUTAGES:
        raise errors.ApiOutageError(
            f"the {domain} API download service is not working right now. "
            "The data is still available with --backend s3.",
            domain=domain,
            detail=API_OUTAGES[domain],
        )
