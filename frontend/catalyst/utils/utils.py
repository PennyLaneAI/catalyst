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

"""Utility functions."""

import inspect
import os
import contextlib
import typing


def are_params_annotated(f: typing.Callable):
    """Return true if all parameters are typed-annotated."""
    signature = inspect.signature(f)
    parameters = signature.parameters
    return all(p.annotation is not inspect.Parameter.empty for p in parameters.values())


def get_type_annotations(func: typing.Callable):
    """Get all type annotations if all parameters are typed-annotated."""
    params_are_annotated = are_params_annotated(func)
    if params_are_annotated:
        return getattr(func, "__annotations__", {}).values()

    return None


@contextlib.contextmanager
def pushd(new_dir):
    """Push a new working directory."""
    cwd = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(cwd)
