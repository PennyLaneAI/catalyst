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

"""
Tracing module.
"""

from catalyst.utils.exceptions import CompileError


class TracingContext:
    """Utility class used for tracing.

    It is used to determine whether the program is currently tracing or not.
    """

    _is_tracing = False

    def __enter__(self):
        assert not TracingContext._is_tracing, "Cannot nest tracing contexts."
        TracingContext._is_tracing = True

    def __exit__(self, *args, **kwargs):
        TracingContext._is_tracing = False

    @staticmethod
    def is_tracing():
        """Returns true or false depending on whether the execution is currently being
        traced.
        """
        return TracingContext._is_tracing

    @staticmethod
    def check_is_tracing(msg):
        """Assert if the execution is currently not being traced.

        Raises: CompileError
        """
        if not TracingContext.is_tracing():
            raise CompileError(msg)

    @staticmethod
    def check_is_not_tracing(msg):
        """Assert if the execution is currently being traced.

        Raises: CompileError
        """
        if TracingContext.is_tracing():
            raise CompileError(msg)
