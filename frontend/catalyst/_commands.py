##############################################################################
# The following file                                                         #
# was taken from the MQT Core project:                                       #
# https://github.com/munich-quantum-toolkit/core                             #
# Small alteration were made for Catalyst                                    #
# The MQT Core project has the following license: MIT License                #
#                                                                            #
# https://github.com/munich-quantum-toolkit/core/blob/main/src/mqt/core/     #
#   _commands.py                                                             #
##############################################################################

"""Useful commands for obtaining information about catalyst."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def include_dir() -> Path:
    """Return the path to the catalyst include directory.

    Raises:
        FileNotFoundError: If the include directory is not found.
        ImportError: If catalyst is not installed.
    """
    try:
        dist = distribution("catalyst")
        located_include_dir = Path(dist.locate_file("catalyst/include"))
        if located_include_dir.exists() and located_include_dir.is_dir():
            return located_include_dir
        msg = "catalyst include files not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "catalyst not installed, installation required to access the include files."
        raise ImportError(msg) from None
