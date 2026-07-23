"""The S3 direct backend for the wave hindcast.

Reads a grid node's record straight from the published .h5 files in the
public bucket. No API and no key. :mod:`.bucket` lists what the bucket
holds and :mod:`.backend` does the fetching.
"""
