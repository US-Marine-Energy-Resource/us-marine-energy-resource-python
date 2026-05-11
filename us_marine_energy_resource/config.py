"""
Default configuration for the US Marine Energy Resource library.

Only storage/access settings are included here. Dataset-specific metadata
lives in the manifest files themselves.
"""

config = {
    "storage": {
        # S3 bucket and prefix for the public OpenEI data store
        "s3_bucket": "marine-energy-data",
        "s3_prefix": "us-tidal",
        # HPC base path for local filesystem access (NLR Kestrel).
        # Set via environment or override when calling query functions.
        "hpc_base_path": "/projects/hindcastra/Tidal/datasets/high_resolution_tidal_hindcast",
    }
}
