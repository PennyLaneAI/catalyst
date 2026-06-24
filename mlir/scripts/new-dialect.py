#!/usr/bin/env python3

# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Add a new MLIR dialect to Catalyst
==================================

This tool adds a new MLIR dialect to Catalyst.

Example
-------

Create a new dialect called 'Awesome':

```console
$ ./new-dialect.py Awesome
🎉 Done! Your new 'Awesome' dialect is now defined in:
.../catalyst/mlir/include/Awesome
.../catalyst/mlir/lib/Awesome

👉 To enable your new dialect, add the following line to the CMakeLists.txt files under mlir/include and mlir/lib:
add_subdirectory(Awesome)
$ cd <catalyst-root-dir>/mlir
$ tree include/Awesome lib/Awesome
include/Awesome
├── CMakeLists.txt
├── IR
│   ├── AwesomeDialect.h
│   ├── AwesomeDialect.td
│   ├── AwesomeOps.h
│   ├── AwesomeOps.td
│   └── CMakeLists.txt
└── Transforms
    └── CMakeLists.txt
lib/Awesome
├── CMakeLists.txt
├── IR
│   ├── AwesomeDialect.cpp
│   ├── AwesomeOps.cpp
│   └── CMakeLists.txt
└── Transforms
    ├── awesome_to_llvm.cpp
    ├── CMakeLists.txt
    └── ConversionPatterns.cpp
"""

import argparse
import datetime
import os
import pathlib
import subprocess
import shutil
import sys
import re


# Placeholder strings in the template files
PLACEHOLDER_DIALECT_NAME = "NewDialect"
PLACEHOLDER_DIALECT_NAMESPACE = "new_dialect"
PLACEHOLDER_DIALECT_LIB = "new-dialect"
PLACEHOLDER_DIALECT_INC_GUARD = "NEW_DIALECT"
PLACEHOLDER_PATTERN = "@@@"


def main():
    """Main entry point to program."""
    try:
        args = parse_args()

        catalyst_project_root = get_catalyst_project_root()
        create_dialect_files(args.name, catalyst_project_root, args)

        names = {
            "name": args.name,
            "namespace": args.namespace,
            "include_guard": _to_snake_case(args.name).upper(),
            "lib": args.lib_name,
        }
        rename_dialect_from_template(args.name, catalyst_project_root, names, args)

        final_check_and_info_message(catalyst_project_root, args)

    except KeyboardInterrupt:
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=_docstring(__doc__))
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print verbose messages; multiple -v result in more verbose messages.",
    )
    parser.add_argument("name", help="The name of the new dialect in CamelCase, e.g. 'MyDialect'.")
    parser.add_argument(
        "--namespace",
        help=(
            "The name of the new dialect namespace in snake_case, e.g. 'my_dialect'. "
            "By default, the new dialect namespace is translated to snake case directly "
            "from the dialect name."
        ),
    )
    parser.add_argument(
        "--lib-name",
        help=(
            "The name of the new dialect library in kebab-case, e.g. 'my-dialect'. "
            "By default, the new dialect library name is translated to kebab case directly "
            "from the dialect name."
        ),
    )

    args = parser.parse_args()

    if args.namespace is None:
        args.namespace = _to_snake_case(args.name)

    if args.lib_name is None:
        args.lib_name = _to_kebab_case(args.name)

    return args


def _docstring(docstring: str):
    """Return summary of docstring"""
    return " ".join(docstring.split("\n")[4:5]) if docstring else ""


def get_catalyst_project_root() -> pathlib.PurePath:
    """Get the absolute path to the root directory of the git repository.

    Returns the path as a ``pathlib.Path`` object. Raises a Runtime error if not currently in a git
    repository or if git is not installed.
    """
    try:
        # Run the git command to get the top-level directory
        result = subprocess.run(
            ["/usr/bin/env", "git", "rev-parse", "--show-toplevel"],
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode as text (UTF-8)
            check=True,  # Raise an error if the command fails
            cwd=pathlib.Path(
                __file__
            ).parent.resolve(),  # Start from the directory containing this script
        )

        # The output is the path, strip any trailing newline
        path = pathlib.Path(result.stdout.strip())

        assert path.exists(), (
            f"The Catalyst project root directory was found to be '{path}', "
            f"but this path does not exist."
        )
        return path

    except subprocess.CalledProcessError as exc:
        # This error occurs if the command fails (e.g., not a git repo)
        raise RuntimeError("Not a git repository") from exc

    except FileNotFoundError as exc:
        # This error occurs if git is not installed or not in PATH
        raise RuntimeError("'git' command not found") from exc


def create_dialect_files(
    dialect_name: str, project_root: pathlib.PurePath, args: argparse.Namespace
):
    """Create the dialect files for the new dialect by copying over the template files to Catalyst.

    Args:
        dialect_name (str): The name of the dialect in CamelCase, e.g. 'MyDialect'.
        project_root (pathlib.PurePath): Catalyst project root directory.
        args (argparse.Namespace): Additional command-line arguments.
    """
    _check_dialect_not_already_exists(project_root, dialect_name)

    templates_path = _get_templates_path()

    # Destination path for copying
    catalyst_mlir_path = project_root / "mlir"

    if args.verbose:
        print(f"Copying template dialect files to {catalyst_mlir_path}")

    # Recursively copy 'include' and 'lib' template files to Catalyst
    shutil.copytree(
        src=templates_path / "include" / PLACEHOLDER_DIALECT_NAME,
        dst=catalyst_mlir_path / "include" / dialect_name,
    )
    shutil.copytree(
        src=templates_path / "lib" / PLACEHOLDER_DIALECT_NAME,
        dst=catalyst_mlir_path / "lib" / dialect_name,
    )

    # Rename files
    _rename_dialect_files(catalyst_mlir_path / "include" / dialect_name, dialect_name)
    _rename_dialect_files(catalyst_mlir_path / "lib" / dialect_name, dialect_name)

    # Special case for new_dialect_to_llvm.cpp
    _rename_dialect_files(
        catalyst_mlir_path / "lib" / dialect_name,
        _to_snake_case(dialect_name),
        old_pattern=PLACEHOLDER_DIALECT_NAMESPACE,
    )


def rename_dialect_from_template(
    dialect_name: str,
    project_root: pathlib.PurePath,
    names: dict[str, str],
    args: argparse.Namespace,
):
    """Rename the placeholder text in the new dialect files according to the new dialect name.

    This function also updates the year in the copyright notices to the current year.

    Args:
        dialect_name (str): The name of the dialect in CamelCase, e.g. 'MyDialect'.
        project_root (pathlib.PurePath): Catalyst project root directory.
        names (dict[str, str]): Dictionary containing dialect names in various forms, e.g. in
            CamelCase for most uses, in snake_case for namespaces, CAPS_SNAKE_CASE for include
            guards, and kebab-case for library names.
        args (argparse.Namespace): Additional command-line arguments.
    """

    dialect_inc_dir = project_root / "mlir" / "include" / dialect_name
    dialect_lib_dir = project_root / "mlir" / "lib" / dialect_name

    assert (
        dialect_inc_dir.exists()
    ), f"No dialect files found for '{dialect_name}' at {dialect_inc_dir}"
    assert (
        dialect_lib_dir.exists()
    ), f"No dialect files found for '{dialect_name}' at {dialect_lib_dir}"

    if args.verbose:
        print(f"Replacing placeholder text with new dialect name '{dialect_name}'")

    # fmt: off
    _run_sed_command(dialect_inc_dir, _placeholder_str(PLACEHOLDER_DIALECT_NAME), names["name"])
    _run_sed_command(dialect_inc_dir, _placeholder_str(PLACEHOLDER_DIALECT_NAMESPACE), names["namespace"])
    _run_sed_command(dialect_inc_dir, _placeholder_str(PLACEHOLDER_DIALECT_INC_GUARD), names["include_guard"])
    _run_sed_command(dialect_inc_dir, _placeholder_str("year"), _get_current_year())

    _run_sed_command(dialect_lib_dir, _placeholder_str(PLACEHOLDER_DIALECT_NAME), names["name"])
    _run_sed_command(dialect_lib_dir, _placeholder_str(PLACEHOLDER_DIALECT_NAMESPACE), names["namespace"])
    _run_sed_command(dialect_lib_dir, _placeholder_str(PLACEHOLDER_DIALECT_INC_GUARD), names["include_guard"])
    _run_sed_command(dialect_lib_dir, _placeholder_str(PLACEHOLDER_DIALECT_LIB), names["lib"])
    _run_sed_command(dialect_lib_dir, _placeholder_str("year"), _get_current_year())
    # fmt: on


def _rename_dialect_files(
    path: str | pathlib.PurePath, dialect_name: str, old_pattern: str = PLACEHOLDER_DIALECT_NAME
):
    """Rename template dialect files found under `path` according to the new dialect name."""
    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            if old_pattern in filename:

                # Create the full old and new paths
                old_path = pathlib.Path(dirpath) / filename
                new_filename = filename.replace(old_pattern, dialect_name)
                new_path = pathlib.Path(dirpath) / new_filename

                # Safety check: make sure the new file name doesn't already exist
                if new_path.exists():
                    print(
                        f"Warning: Attempting to rename file to '{new_path}', "
                        f"but it already exists; skipping."
                    )
                else:
                    try:
                        # Rename the file
                        old_path.rename(new_path)

                    except OSError as exc:
                        raise RuntimeError(
                            f"Could not rename file '{old_path}' to '{new_path}'"
                        ) from exc


def _check_dialect_not_already_exists(catalyst_project_root: pathlib.PurePath, dialect_name: str):
    """Check that a dialect with the given name does not already exist in Catalyst.

    Raises a RuntimeError if one with the same name does already exist.
    """

    def _raise_msg(candidate_path, dialect_name):
        raise RuntimeError(
            f"There is already a dialect '{dialect_name}' defined at {candidate_path}"
        )

    if (candidate_path := catalyst_project_root / "mlir" / "include" / dialect_name).exists():
        _raise_msg(candidate_path, dialect_name)

    if (candidate_path := catalyst_project_root / "mlir" / "lib" / dialect_name).exists():
        _raise_msg(candidate_path, dialect_name)


def _get_templates_path() -> pathlib.PurePath:
    """Returns the path of directory containing the dialect templates files."""
    # Get template dir relative to current script; return as absolute path
    this_script_path = pathlib.Path(__file__).resolve()
    templates_path = this_script_path.parent / "dialect-templates"

    return templates_path


def _run_sed_command(path: str | pathlib.PurePath, search_pattern: str, replace_str: str):
    """Run the sed command in a subprocess that replaces `search_pattern` with `replace_str` in all
    files contained under `path`.
    """
    if isinstance(path, pathlib.PurePath):
        path = str(path)

    find_command = ["find", path, "-type", "f", "-print0"]

    with subprocess.Popen(find_command, stdout=subprocess.PIPE, text=True) as pipe:
        sed_command = ["xargs", "-0", "sed", "-i", f"s/{search_pattern}/{replace_str}/g"]
        subprocess.run(
            sed_command,
            stdin=pipe.stdout,  # Pipe previous output to this one
            text=True,  # Decode as text (UTF-8)
            check=True,  # Raise an error if the command fails
        )


def _placeholder_str(pattern: str) -> str:
    """Helper function that returns the placeholder text for the given `pattern` as it appears in
    the template files.

    Example:
    >>> _placeholder_str("NewDialect")
    '@@@NewDialect@@@'
    """
    return f"{PLACEHOLDER_PATTERN}{pattern}{PLACEHOLDER_PATTERN}"


def _get_current_year() -> str:
    """Returns the current year as a string."""
    return str(datetime.datetime.now().year)


def _to_snake_case(name: str) -> str:
    """
    Convert a CamelCase string (e.g., 'MyNewType') to snake_case (e.g., 'my_new_type').

    This version correctly handles acronyms. e.g., "SomeHTTPData" -> "some_http_data"
    """
    # 1. Find lowercase/digit followed by uppercase: 'myNew' -> 'my_New'
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)

    # 2. Find uppercase followed by uppercase-lowercase: 'HTMLParser' -> 'HTML_Parser'
    s2 = re.sub(r"([A-Z])([A-Z][a-z])", r"\1_\2", s1)

    # 3. Convert the whole string to lowercase
    return s2.lower()


def _to_kebab_case(name: str) -> str:
    """
    Convert a CamelCase string (e.g., 'MyNewType') to kebab-case (e.g., 'my-new-type').

    This version correctly handles acronyms. e.g., "SomeHTTPData" -> "some-http-data"
    """
    # 1. Find lowercase/digit followed by uppercase: 'myNew' -> 'my_New'
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)

    # 2. Find uppercase followed by uppercase-lowercase: 'HTMLParser' -> 'HTML_Parser'
    s2 = re.sub(r"([A-Z])([A-Z][a-z])", r"\1-\2", s1)

    # 3. Convert the whole string to lowercase
    return s2.lower()


def final_check_and_info_message(project_root: pathlib.PurePath, args: argparse.Namespace):
    """Perform final validation on the newly created dialect files and print info message with
    summary and further instructions.

    Args:
        project_root (pathlib.PurePath): Catalyst project root directory.
        args (argparse.Namespace): Additional command-line arguments.
    """
    dialect_inc_dir = project_root / "mlir" / "include" / args.name
    dialect_lib_dir = project_root / "mlir" / "lib" / args.name

    for dialect_dir in [dialect_inc_dir, dialect_lib_dir]:
        assert dialect_dir.exists(), f"The directory '{dialect_dir}' should exist but it does not."

    print(
        f"🎉 Done! Your new '{args.name}' dialect is now defined in:\n"
        f"{dialect_inc_dir}\n"
        f"{dialect_lib_dir}"
    )

    print(
        f"\n"
        f"👉 To enable your new dialect, add the following line to the CMakeLists.txt files under "
        f"mlir/include and mlir/lib:\n"
        f"add_subdirectory({args.name})"
    )


if __name__ == "__main__":
    sys.exit(main())
