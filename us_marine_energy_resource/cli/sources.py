"""Marine Energy Resource Data Source Specification"""

from __future__ import annotations

from ._display import console
from ._links import _link

_SOURCES_HELP = "List the marine energy resource data sources"

# One entry per dataset: the MHKDR submission that publishes it, a browser view
# of the bucket, and the s3:// location the other commands read.
_SOURCES: tuple[dict[str, str], ...] = (
    {
        "name": "tidal",
        "title": "U.S. DOE H2O High Resolution Tidal Hindcast",
        "submission": "https://mhkdr.openei.org/submissions/632",
        "browser": "https://data.openei.org/s3_viewer?bucket=marine-energy-data&prefix=us-tidal%2F",
        "s3": "s3://marine-energy-data/us-tidal/",
    },
    {
        "name": "wave",
        "title": "U.S. DOE H2O High-Resolution Wave Hindcast",
        "submission": "https://mhkdr.openei.org/submissions/326",
        "browser": "https://data.openei.org/s3_viewer?bucket=wpto-pds-us-wave",
        "s3": "s3://wpto-pds-us-wave/",
    },
)


def sources() -> None:
    """Print each dataset's submission page, browser view, and S3 location."""
    for entry in _SOURCES:
        console.print(f"[bright_blue]{entry['name']}[/]  {entry['title']}")
        console.print(f"  submission  {_link(entry['submission'])}", highlight=False)
        console.print(f"  browser     {_link(entry['browser'])}", highlight=False)
        console.print(f"  s3          {entry['s3']}", highlight=False)
        console.print()
