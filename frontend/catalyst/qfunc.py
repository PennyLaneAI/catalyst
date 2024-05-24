# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

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
This module contains a patch for the upstream qml.QNode behaviour, in particular around
what happens when a QNode object is called during tracing. Mostly this involves bypassing
the default behaviour and replacing it with a function-like "QNode" primitive.
"""

import pennylane as qml
from jax.core import eval_jaxpr
from jax.tree_util import tree_flatten, tree_unflatten

from catalyst.device import (
    BackendInfo,
    QJITDevice,
    QJITDeviceNewAPI,
    extract_backend_info,
    validate_device_capabilities,
)
from catalyst.jax_extras import (
    deduce_avals,
    get_implicit_and_explicit_flat_args,
    unzip2,
)
from catalyst.jax_primitives import func_p
from catalyst.jax_tracer import trace_quantum_function
from catalyst.utils.toml import (
    DeviceCapabilities,
    ProgramFeatures,
    get_device_capabilities,
)


class QFunc:
    """A device specific quantum function.

    Args:
        qfunc (Callable): the quantum function
        shots (int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values
        device (a derived class from QubitDevice): a device specification which determines
            the valid gate set for the quantum function
    """

    def __new__(cls):
        raise NotImplementedError()  # pragma: no-cover

    @staticmethod
    def extract_backend_info(
        device: qml.QubitDevice, capabilities: DeviceCapabilities
    ) -> BackendInfo:
        """Wrapper around extract_backend_info in the runtime module."""
        return extract_backend_info(device, capabilities)

    # pylint: disable=no-member, attribute-defined-outside-init
    def __call__(self, *args, **kwargs):
        assert isinstance(self, qml.QNode)

        # TODO: Move the capability loading and validation to the device constructor when the
        # support for old device api is dropped.
        program_features = ProgramFeatures(self.device.shots is not None)
        device_capabilities = get_device_capabilities(self.device, program_features)
        backend_info = QFunc.extract_backend_info(self.device, device_capabilities)

        # Validate decive operations against the declared capabilities
        validate_device_capabilities(self.device, device_capabilities)

        if isinstance(self.device, qml.devices.Device):
            qjit_device = QJITDeviceNewAPI(self.device, device_capabilities, backend_info)
        else:
            qjit_device = QJITDevice(
                self.device, device_capabilities, self.device.shots, self.device.wires, backend_info
            )

        def _eval_quantum(*args):
            closed_jaxpr, out_type, out_tree = trace_quantum_function(
                self.func, qjit_device, args, kwargs, qnode=self
            )
            args_expanded = get_implicit_and_explicit_flat_args(None, *args)
            res_expanded = eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args_expanded)
            _, out_keep = unzip2(out_type)
            res_flat = [r for r, k in zip(res_expanded, out_keep) if k]
            return tree_unflatten(out_tree, res_flat)

        flattened_fun, _, _, out_tree_promise = deduce_avals(_eval_quantum, args, {})
        args_flat = tree_flatten(args)[0]
        res_flat = func_p.bind(flattened_fun, *args_flat, fn=self)
        return tree_unflatten(out_tree_promise(), res_flat)[0]
