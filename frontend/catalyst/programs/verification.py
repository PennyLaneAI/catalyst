# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains for program verification.
"""

from dataclasses import dataclass
from typing import List, Optional, Callable

from pennylane.tape import QuantumTape
from pennylane.operation import Operation

from catalyst.jax_extras import DynamicJaxprTrace
from catalyst.utils.exceptions import CompileError
from catalyst.utils.toml import DeviceCapabilities
from catalyst.tracing.contexts import EvaluationContext
from catalyst.utils.exceptions import DifferentiableCompileError


def verify_inverses(device: "AnyQJITDevice", tape: QuantumTape) -> None:
    """Verify quantum program against the device capabilities.

    Raises: CompileError
    """
    pass

def verify_control(device: "AnyQJITDevice", tape: QuantumTape) -> None:
    """Verify quantum program against the device capabilities.

    Raises: CompileError
    """
    pass


def _is_differentiable_on_device(op_name:str, device: "AnyQJITDevice") -> bool:
    """Checks if the operation `op_name` is differentiable on the `device`"""
    if op_name not in device.capabilities.native_ops:
        return False
    return device.capabilities.native_ops[op_name].differentiable


def _verify_differentiability(
    device: "AnyQJITDevice",
    tape: QuantumTape,
    ctx: Optional[EvaluationContext] = None,
) -> None:
    """Verify differentiability recursively. """

    # FIXME: How should we re-organize the code to avoid this kind of circular dependency.
    # Another candidate: `from catalyst.qjit_device import AnyQJITDevice`
    from catalyst.jax_tracer import has_nested_tapes, nested_quantum_regions

    ctx = ctx if ctx is not None else EvaluationContext.get_main_tracing_context()
    ops2 = []
    for op in tape.operations:
        if has_nested_tapes(op):
            for region in nested_quantum_regions(op):
                with EvaluationContext.frame_tracing_context(ctx, region.trace) as nested_ctx:
                    _verify_differentiability(
                        device, region.quantum_tape, nested_ctx
                    )

        if not _is_differentiable_on_device(op.name, device):
            raise DifferentiableCompileError(f'{op.name} is non-differentiable on {device.name}')


def verify_differentiability(device: "AnyQJITDevice", tape: QuantumTape) -> None:
    """Verify quantum program against the device capabilities.

    Raises: DifferentiableCompileError
    """
    _verify_differentiability(device, tape)

