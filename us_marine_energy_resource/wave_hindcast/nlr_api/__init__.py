"""The NLR developer download API backend for the wave hindcast.

The default way to fetch a grid node's record: :mod:`.client` talks to the
API, :mod:`.archive` waits for, downloads, and organizes the archives the
service builds, and :mod:`.backend` ties the two together. Credentials are
``NLR_DEVELOPER_API_KEY`` and ``NLR_DEVELOPER_EMAIL``, looked up in the
environment, then a ``.env`` file in the current directory, then
``~/.mer.env``, with a free key from https://developer.nlr.gov/signup/.
Nothing here prints: progress lands on the ``on_event`` callback the caller
provides, so the CLI can render it and library use stays silent.
"""
