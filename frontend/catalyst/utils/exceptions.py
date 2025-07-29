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
"""Custom Catalyst exceptions."""


class AutoGraphError(Exception):
    """Errors related to Catalyst's AutoGraph module."""


class CompileError(Exception):
    """Error encountered in the compilation phase."""


class DifferentiableCompileError(CompileError):
    """An error indicating an invalid differentiation configuration."""


class CompatibilityError(Exception):
    """Error raised when Catalyst control flow is used with PennyLane capture enabled.

    This error indicates that Catalyst's control flow functions (for_loop, while_loop, cond)
    cannot be used in contexts where PennyLane's capture mode is active. Users should use
    the corresponding PennyLane functions instead for compatibility with capture functionality.
    """

    def __init__(self, control_flow_type="control flow", message=None):
        if message is None:
            if control_flow_type == "for_loop":
                message = (
                    "catalyst.for_loop is not supported with PennyLane's capture feature enabled. "
                    "For compatibility with program capture, please use qml.for_loop instead. "
                    "See the documentation for more information on using qml.for_loop with "
                    "captured quantum programs."
                )
            elif control_flow_type == "while_loop":
                message = (
                    "catalyst.while_loop is not supported with PennyLane's capture feature enabled. "
                    "For compatibility with program capture, please use qml.while_loop instead. "
                    "See the documentation for more information on using qml.while_loop with "
                    "captured quantum programs."
                )
            elif control_flow_type == "cond":
                message = (
                    "catalyst.cond is not supported with PennyLane's capture feature enabled. "
                    "For compatibility with program capture, please use qml.cond instead. "
                    "See the documentation for more information on using qml.cond with "
                    "captured quantum programs."
                )
            else:
                message = (
                    f"catalyst.{control_flow_type} is not supported with PennyLane's capture feature enabled. "
                    "For compatibility with program capture, please use the corresponding PennyLane "
                    "function instead. See the documentation for more information."
                )
        super().__init__(message)
