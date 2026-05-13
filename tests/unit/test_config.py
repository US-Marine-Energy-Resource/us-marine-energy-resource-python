"""Simple smoke test for the default config."""

from us_marine_energy_resource.config import config


def test_config_has_expected_keys() -> None:
    """config dict exposes the required storage keys."""
    storage = config["storage"]
    assert storage["s3_bucket"] == "marine-energy-data"
    assert storage["s3_prefix"] == "us-tidal"
    assert "hpc_base_path" in storage
