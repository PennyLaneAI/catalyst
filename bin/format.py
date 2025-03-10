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
import re
import subprocess
import sys

from utils import get_cpp_files

CLANG_FMT_BIN = "clang-format"

IGNORE_PATTERNS = ["external", "build", "llvm-project", "mlir-hlo", "Enzyme"]

DEFAULT_CLANG_FORMAT_VERSION = 13

CLANG_FMT_CNFG_PATH = "../.clang-format"

BASE_ARGS = f"-assume-filename={CLANG_FMT_CNFG_PATH}"


def parse_version(version_string):
    version_rgx = r"version (\d+)"

    m = re.search(version_rgx, version_string)
    return int(m.group(1))


def check_bin(command):
    try:
        p = subprocess.run(
            [command, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        version = parse_version(p.stdout)

        if version < DEFAULT_CLANG_FORMAT_VERSION:
            print(
                f"Using clang-format version {version}. \
                    As this is lower than the version used for the CI, \
                    the CI may fail even after formatting."
            )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{command} is not installed or is not in PATH.") from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Opinionated C/C++ formatter. Based on clang-format"
    )
    parser.add_argument(
        "paths", nargs="+", metavar="DIR", help="paths to the root source directories"
    )
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="don't write files, just return status. "
        "A non-zero return code indicates some files would be re-formatted",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print detailed information about format violations",
    )
    parser.add_argument(
        "-f",
        "--cfversion",
        type=int,
        default=0,
        action="store",
        help="set a version number for clang-format",
    )
    return parser.parse_args()


def fmt(args, command) -> int:
    files = get_cpp_files(args.paths, ignore_patterns=IGNORE_PATTERNS)
    cmd = (command, BASE_ARGS, "-i", *files)

    sys.stderr.write(f"Formatting {len(files)} files in {args.paths}.\n")

    ret = subprocess.run(cmd, capture_output=True, universal_newlines=True)
    if ret.returncode != 0:
        sys.stderr.write(ret.stderr)
        return 1

    return 0


def check(args, command) -> int:
    cmd = (command, BASE_ARGS, "--dry-run", "-Werror")

    needs_reformatted_ct = 0
    files = get_cpp_files(args.paths, ignore_patterns=IGNORE_PATTERNS)

    for src_file in files:
        ret = subprocess.run((*cmd, src_file), capture_output=True, universal_newlines=True)

        if ret.returncode != 0:
            sys.stderr.write(f"Error: {src_file} would be reformatted.\n")
            if args.verbose:
                sys.stderr.write(ret.stderr)

            needs_reformatted_ct += 1

    sys.stderr.write(f"{needs_reformatted_ct} files would be re-formatted.\n")
    sys.stderr.write(f"{len(files) - needs_reformatted_ct} would be left unchanged.\n")

    return needs_reformatted_ct


if __name__ == "__main__":
    args = parse_args()

    cf_version = args.cfversion
    cf_cmd = "clang-format"

    if cf_version:
        cf_cmd += f"-{cf_version}"

    check_bin(cf_cmd)

    if args.check:
        ret = check(args, cf_cmd)
    else:
        ret = fmt(args, cf_cmd)

    sys.exit(int(ret > 0))
