# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Functions to interface with the filesystem.
"""

import pathlib
import shutil
import sys
import tempfile


class Directory:
    """Abstracts over pathlib and tempfile."""

    def __init__(self, pathlibOrTempfile):
        is_tempdir = isinstance(pathlibOrTempfile, tempfile.TemporaryDirectory)
        is_path = isinstance(pathlibOrTempfile, pathlib.Path)
        is_valid = is_tempdir or is_path
        assert is_valid, "Invalid initialization of Directory"
        self._impl = pathlibOrTempfile

    def __str__(self):
        if isinstance(self._impl, tempfile.TemporaryDirectory):
            return self._impl.name
        return str(self._impl)

    def is_dir(self):
        """Returns True if it is a directory.

        Should always return True for both, however, we leave it to the implementation to actually
        confirm it.
        """
        if isinstance(self._impl, tempfile.TemporaryDirectory):
            return True
        return self._impl.is_dir()

    def cleanup(self):
        """Remove the contents of the directory."""
        if isinstance(self._impl, tempfile.TemporaryDirectory):
            # The temporary directory can clean up
            # after itself...
            return
        shutil.rmtree(str(self))


class TemporaryDirectorySilent(tempfile.TemporaryDirectory):
    """Derived class from tempfile.TemporaryDirectory

    This overrides the _cleanup method which would normally emit a warning
    after removing the directory. This warning is unconditional and it is emitted
    because it is not called explicitly.
    """

    @classmethod
    def _cleanup(cls, name, warn_message, **kwargs):  # pylint: disable=arguments-differ
        """Ignore ResourceWarning during cleanup."""
        del warn_message
        minor_version = sys.version_info[1]
        if kwargs.get("delete") and minor_version >= 12:
            # Changed in version 3.12: Removed the "delete" kwargs
            del kwargs["delete"]  # pragma: nocover
        # pylint: disable-next=protected-access
        tempfile.TemporaryDirectory._rmtree(name, **kwargs)


class WorkspaceManager:
    """Singleton object that manages the output files for the IR.

    A good motivation for this is whether or not we are allowed to overwrite
    folders in the user's directory if they match the preferred name.

    This happens whenever we want to compile a function with the same name
    multiple times and we have used the user facing option keep_intermediates=True.
    """

    # Operating System agnostic way of finding out what is the temporary
    # directory. See https://docs.python.org/3.10/library/tempfile.html#tempfile.gettempdir
    tempdir = pathlib.Path(tempfile.gettempdir())

    @staticmethod
    def get_or_create_workspace(name, path=None):
        """
        Args:
            name (str): Directory name
            path (Optional(str)): Directory path
                                If path is None, then it will be a temporary directory
                                stored in whatever temporary directory is specific to the
                                operating system.
        """
        path, name = WorkspaceManager._get_preferred_abspath(name, path)
        return Directory(WorkspaceManager._get_or_create_directory(path, name))

    @staticmethod
    def _get_preferred_abspath(name, path=None):
        preferred_path = pathlib.Path(path) if path is not None else WorkspaceManager.tempdir
        preferred_name = pathlib.Path(name)
        return preferred_path, preferred_name

    @staticmethod
    def _get_or_create_directory(path, name):
        if path == WorkspaceManager.tempdir:
            # TODO: Once Python 3.12 becomes the least supported version of python, consider
            # setting the fields: delete and delete_on_close.
            # This can likely avoid having all the code below.
            return TemporaryDirectorySilent(dir=path.resolve(), prefix=name.name)

        count = 1
        curr_preferred_abspath = path / name
        preferred_name = name.name

        # TODO: Maybe just look for the last one?
        while curr_preferred_abspath.exists():
            curr_preferred_name_str = preferred_name + "_" + str(count)
            curr_preferred_name = pathlib.Path(curr_preferred_name_str)
            curr_preferred_abspath = path / curr_preferred_name
            count += 1

        free_preferred_abspath = pathlib.Path(curr_preferred_abspath)
        free_preferred_abspath.mkdir()
        return free_preferred_abspath
