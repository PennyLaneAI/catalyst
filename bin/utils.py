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

from fnmatch import fnmatch
from pathlib import Path
from re import compile as re_compile

SRCFILE_EXT = ["c", "cc", "cpp", "cxx", "cu"]
HEADERFILE_EXT = ["h", "hh", "hpp", "hxx", "cuh"]

rgx_gitignore_comment = re_compile("#.*$")


def get_cpp_files_from_path(path, ignore_patterns=None, use_gitignore=True, header_only=False):
    """return set of C++ source files from a path

    Args:
        paths (pathlib.Path or str): a path to process
        ignore_patterns: patterns to ignore
        use_gitignore: find ignore patterns from .gitignore
        header_only: find only header files when true
    """
    path = Path(path)
    files_rel = set()  # file paths relative to path

    exts = HEADERFILE_EXT
    if not header_only:
        exts += SRCFILE_EXT
    for ext in exts:
        for file_path in path.rglob(f"*.{ext}"):
            files_rel.add(file_path.relative_to(path))

    if ignore_patterns is None:
        ignore_patterns = []

    if use_gitignore:
        # simple gitignore parser
        gitignore_file = path.joinpath(".gitignore")
        if gitignore_file.exists():
            with gitignore_file.open() as f:
                for line in f.readlines():
                    line = rgx_gitignore_comment.sub("", line)
                    line = line.strip()
                    if line:
                        ignore_patterns.append(line)

    files_to_remove = set()
    for ignore_pattern in ignore_patterns:
        for f in files_rel:
            if fnmatch(str(f), ignore_pattern) or ignore_pattern in str(f):
                files_to_remove.add(f)

    files_rel -= files_to_remove

    return set(str(path.joinpath(f)) for f in files_rel)


def get_cpp_files(paths, ignore_patterns=None, use_gitignore=True, header_only=False):
    """return list of C++ source files from paths.

    Args:
        paths (list): list of all paths to process
        ignore_patterns: patterns to ignore
        use_gitignore: find ignore patterns from .gitignore
        header_only: find only header files when true
    """
    files = set()
    for path in paths:
        files |= get_cpp_files_from_path(path, ignore_patterns, use_gitignore, header_only)
    return list(files)
