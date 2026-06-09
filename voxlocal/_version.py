# SPDX-License-Identifier: MIT
__version__ = "0.1.0"

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("voxlocal")
except Exception:  # noqa: S110
    # Fallback for development editable installs
    pass

__all__ = ["__version__"]
