"""External URLs referenced in CLI help text."""

DATASET_CITATION = "https://mhkdr.openei.org/submissions/632"
DOCS = "https://github.com/US-Marine-Energy-Resource/us-marine-energy-resource-python"
ISSUES = f"{DOCS}/issues"
S3_BROWSER = "https://data.openei.org/s3_viewer?bucket=marine-energy-data&prefix=us-tidal%2F"
GEOJSON_TOOL = "https://geojson.io/next/"

_S3_BUCKET = "marine-energy-data"
_S3_PREFIX = "us-tidal"
_S3_REGION = "us-west-2"


def _link(url: str) -> str:
    """Wrap a URL in a Rich hyperlink tag with the URL as display text."""
    return f"[link={url}]{url}[/link]"


def s3_uri(relative_path: str) -> str:
    """Return the s3:// URI for a relative dataset file path."""
    return f"s3://{_S3_BUCKET}/{_S3_PREFIX}/{relative_path}"


def http_url(relative_path: str) -> str:
    """Return the HTTPS URL for a relative dataset file path."""
    return f"https://{_S3_BUCKET}.s3.{_S3_REGION}.amazonaws.com/{_S3_PREFIX}/{relative_path}"
