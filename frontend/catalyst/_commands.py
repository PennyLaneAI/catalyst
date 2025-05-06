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



def cmake_dir() -> Path:
    """Return the path to the catalyst CMake module directory.

    Raises:
        FileNotFoundError: If the CMake module directory is not found.
        ImportError: If catalyst is not installed.
    """
    try:
        dist = distribution("catalyst")
        located_cmake_dir = Path(dist.locate_file("catalyst/cmake"))
        if located_cmake_dir.exists() and located_cmake_dir.is_dir():
            return located_cmake_dir
        msg = "catalyst CMake files not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "catalyst not installed, installation required to access the CMake files."
        raise ImportError(msg) from None
