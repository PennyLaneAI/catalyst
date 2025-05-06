##############################################################################
# The following file                                                         #
# was taken from the MQT Core project:                                       #
# https://github.com/munich-quantum-toolkit/core                             #
# Small alteration were made for Catalyst                                    #
# The MQT Core project has the following license: MIT License                #
#                                                                            #
# https://github.com/munich-quantum-toolkit/core/blob/main/src/mqt/core/     #
#   __main__.py                                                              #
##############################################################################

"""Command line interface for catalyst."""

from __future__ import annotations

import argparse
import sys

from ._commands import cmake_dir
from ._version import __version__


def main() -> None:
    """Entry point for the python module's command line interface of catalyst.

    This function is called when running the catalyst module.

    .. code-block:: bash

        python -m catalyst [--version] [--cmake_dir]

    It provides the following command line options:

    - :code:`--version`: Print the version and exit.
    - :code:`--cmake_dir`: Print the path to the catalyst CMake module directory.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="version", version=__version__, help="Print version and exit.")

    parser.add_argument(
        "--cmake_dir", action="store_true", help="Print the path to the catalyst CMake module directory."
    )
    args = parser.parse_args()
    if not sys.argv[1:]:
        parser.print_help()
    if args.cmake_dir:
        print(cmake_dir().resolve())


if __name__ == "__main__":
    main()
