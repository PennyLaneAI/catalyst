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
from enum import Enum


class EvaluationMode(Enum):
    INTERPRETATION = 0
    CLASSICAL_COMPILATION = 2
    QUANTUM_COMPILATION = 1


class TracingContext:
    """Utility class used for tracing.

    It is used to determine whether the program is currently tracing or not.
    """

    _tracing_stack = []

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        TracingContext._tracing_stack.append(self.state)

    def __exit__(self, *args, **kwargs):
        TracingContext._tracing_stack.pop()

    @staticmethod
    def check_is_valid_push(state):
        mode = TracingContext.get_mode()
        if EvaluationMode.INTERPRETATION == mode:
            return
        if (
            EvaluationMode.CLASSICAL_COMPILATION == mode
            and EvaluationMode.QUANTUM_COMPILATION == state
        ):
            return

        raise CompileError("Invalid state.")

    @staticmethod
    def is_interpretation():
        return not TracingContext._tracing_stack

    @staticmethod
    def is_quantum_compilation():
        return EvaluationMode.QUANTUM_COMPILATION == TracingContext._tracing_stack[-1]

    @staticmethod
    def is_classical_compilation():
        return EvaluationMode.CLASSICAL_COMPILATION == TracingContext._tracing_stack[-1]

    @staticmethod
    def get_mode():
        if not TracingContext._tracing_stack:
            return EvaluationMode.INTERPRETATION
        return TracingContext._tracing_stack[-1]

    @staticmethod
    def is_tracing():
        """Returns true or false depending on whether the execution is currently being
        traced.
        """
        return bool(TracingContext._tracing_stack)

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
