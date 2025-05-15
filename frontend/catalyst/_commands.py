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

from pathlib import Path


def get_include_dir() -> Path:
    """Return the path to Catalyst's include directory.

    Raises:
        FileNotFoundError: If the include directory is not found.
        ImportError: If Catalyst is not installed.
    """
    located_include_dir = Path(__file__).parent/"include"
    if located_include_dir.exists() and located_include_dir.is_dir():
        return located_include_dir
    msg = "Catalyst's include directory not found."
    raise FileNotFoundError(msg)
