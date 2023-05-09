#!/usr/bin/python3

# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import sys

from utils import get_cpp_files

if __name__ == "__main__":
    """
    This program output a json list of all C++ source files.
    """
    parser = argparse.ArgumentParser(description="Output C/C++ files in json list")
    parser.add_argument(
        "--header-only",
        action="store_true",
        dest="header_only",
        help="whether only include header files",
    )
    parser.add_argument(
        "paths", nargs="+", metavar="DIR", help="paths to the root source directories"
    )
    parser.add_argument(
        "--exclude-dirs", dest="exclude_dirs", nargs="*", metavar="DIR", help="paths exclude from"
    )

    args = parser.parse_args()

    files = set(get_cpp_files(args.paths, header_only=args.header_only))
    if args.exclude_dirs:
        files_excludes = set(get_cpp_files(args.exclude_dirs, header_only=args.header_only))
        files -= files_excludes

    json.dump(list(files), sys.stdout)
