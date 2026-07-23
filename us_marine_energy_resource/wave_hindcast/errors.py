"""Exceptions for the wave hindcast client.

One class per failure actually observed against the service, so callers can
branch on the cause rather than parsing message text. The distinctions that
matter in practice:

* a rejected request (`RequestError`) is worth retrying with different inputs
* an accepted-then-failed download (`DownloadError`) is not a client problem
* an `ApiOutageError` means the API backend cannot succeed until the service
  is fixed (the s3 backend still can)

Messages are one line. Detail belongs in the attributes.
"""

from __future__ import annotations

from collections.abc import Iterable


class WaveHindcastError(Exception):
    """Base class for every error raised by this package."""


# -- Local configuration: nothing was sent --


class ConfigurationError(WaveHindcastError):
    """The client is not set up correctly."""


class CredentialsMissingError(ConfigurationError):
    """NLR_DEVELOPER_API_KEY or NLR_DEVELOPER_EMAIL is unset."""


class IndexMissingError(ConfigurationError):
    """The grid-node index could not be found, downloaded, or generated."""


class UnknownSiteError(ConfigurationError):
    """No site is configured under that name."""


# -- Location resolution: offline, before any request --


class PointOutsideDomainError(WaveHindcastError, ValueError):
    """The coordinate falls outside every hindcast domain.

    Parameters
    ----------
    message : str
        One line description.
    lat, lon : float, optional
        The coordinate that failed to resolve.
    domains : iterable of str
        The domains that were checked.
    """

    def __init__(
        self,
        message: str,
        lat: float | None = None,
        lon: float | None = None,
        domains: Iterable[str] = (),
    ) -> None:
        """Record the coordinate and the domains that were checked."""
        super().__init__(message)
        self.lat = lat
        self.lon = lon
        self.domains = tuple(domains)


# -- The API rejected the request --


class RequestError(WaveHindcastError):
    """The API refused the request.

    Parameters
    ----------
    message : str
        One line description.
    status : int, optional
        HTTP status code, when one was received.
    errors : iterable of str
        The service's own ``errors`` array, when it returned one.

    Attributes
    ----------
    status : int or None
        HTTP status code.
    errors : list of str
        The service's own ``errors`` array, when it returned one.
    """

    def __init__(self, message: str, status: int | None = None, errors: Iterable[str] = ()) -> None:
        """Record the HTTP status and the service's own error strings."""
        super().__init__(message)
        self.status = status
        self.errors = list(errors)


class AuthenticationError(RequestError):
    """API key missing, invalid, disabled, unverified, or unauthorized (403)."""


class RateLimitError(RequestError):
    """Rate limit exceeded (429).

    The attribute probe counts as a request.
    """


class EndpointNotFoundError(RequestError):
    """No endpoint at that URL (404), usually a wrong dataset slug."""


class InvalidAttributeError(RequestError):
    """One or more requested attributes are not available for this domain.

    Parameters
    ----------
    message : str
        One line description.
    status : int, optional
        HTTP status code, when one was received.
    errors : iterable of str
        The service's own ``errors`` array, when it returned one.
    valid : iterable of str
        The attributes the endpoint enumerated as acceptable.

    Attributes
    ----------
    valid : list of str
        The attributes the endpoint enumerated as acceptable.
    """

    def __init__(
        self,
        message: str,
        status: int | None = None,
        errors: Iterable[str] = (),
        valid: Iterable[str] = (),
    ) -> None:
        """Record the endpoint's own list of acceptable attributes."""
        super().__init__(message, status, errors)
        self.valid = list(valid)


class InvalidYearError(RequestError):
    """A requested year is outside what this domain serves.

    Atlantic and Hawaii stop at 2010 despite the documentation claiming 2020.
    """


class NoDataAtLocationError(RequestError):
    """The geometry resolved to somewhere the domain has no nodes."""


class QueueFullError(RequestError):
    """The service is shedding load and refused to queue the job."""


# -- The request was accepted, the data never arrived --


class DownloadError(WaveHindcastError):
    """The API accepted the request but no archive resulted.

    Parameters
    ----------
    message : str
        One line description.
    site : str, optional
        The site the download was for.
    url : str, optional
        The download URL involved.
    """

    def __init__(self, message: str, site: str | None = None, url: str | None = None) -> None:
        """Record the site and download URL involved."""
        super().__init__(message)
        self.site = site
        self.url = url


class ApiOutageError(DownloadError):
    """The domain's API download service is recorded as broken upstream.

    Parameters
    ----------
    message : str
        One line description.
    domain : str, optional
        The affected domain.
    detail : str
        What was tried and observed.

    Attributes
    ----------
    domain : str or None
    detail : str
        What was tried and observed.
    """

    def __init__(self, message: str, domain: str | None = None, detail: str = "") -> None:
        """Record the affected domain and what was observed."""
        super().__init__(message)
        self.domain = domain
        self.detail = detail


class ArchiveTimeoutError(DownloadError):
    """The archive was still unavailable when polling gave up.

    The download URL is kept in the manifest, so retrying resumes rather than
    re-requesting.
    """


class ArchiveFailedError(DownloadError):
    """The server failed to build the archive.

    Indicated by a "NLR Data Download Error" email; nothing client-side fixes it.
    """


# -- Data on disk --


class DataError(WaveHindcastError):
    """A problem with data already downloaded."""


class CacheMissError(DataError, FileNotFoundError):
    """Nothing cached for this site. Subclasses FileNotFoundError deliberately."""


class ArchiveCorruptError(DataError):
    """The archive is unreadable or contains no year CSVs."""


class ArchiveUnmatchedError(DataError):
    """An archive could not be matched to a configured site."""
