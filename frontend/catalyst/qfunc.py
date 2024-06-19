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
import logging
from copy import copy
from typing import Callable, Sequence

import jax.numpy as jnp
import pennylane as qml
from jax.core import eval_jaxpr
from jax.tree_util import tree_flatten, tree_unflatten
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    ProbabilityMP,
    SampleMP,
    VarianceMP,
)
from pennylane.transforms.dynamic_one_shot import (
    init_auxiliary_tape,
    parse_native_mid_circuit_measurements,
)

import catalyst
from catalyst.device import (
    BackendInfo,
    QJITDevice,
    QJITDeviceNewAPI,
    extract_backend_info,
    get_device_capabilities,
    get_device_shots,
    validate_device_capabilities,
)
from catalyst.jax_extras import (
    deduce_avals,
    get_implicit_and_explicit_flat_args,
    unzip2,
)
from catalyst.jax_primitives import func_p
from catalyst.jax_tracer import trace_quantum_function
from catalyst.logging import debug_logger
from catalyst.utils.toml import DeviceCapabilities, ProgramFeatures

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _validate_mcm_config(mcm_config, shots):
    """Helper function for validating that the mcm_config is valid for executing."""
    mcm_config.postselect_mode = mcm_config.postselect_mode if shots else None
    if mcm_config.mcm_method is None:
        mcm_config.mcm_method = (
            "one-shot" if mcm_config.postselect_mode == "hw-like" else "single-branch-statistics"
        )
    if mcm_config.mcm_method == "deferred":
        raise ValueError("mcm_method='deferred' is not supported with Catalyst.")
    if (
        mcm_config.mcm_method == "single-branch-statistics"
        and mcm_config.postselect_mode == "hw-like"
    ):
        raise ValueError(
            "Cannot use postselect_mode='hw-like' with Catalyst when mcm_method != 'one-shot'."
        )
    if mcm_config.mcm_method == "one-shot" and shots is None:
        raise ValueError(
            "Cannot use the 'one-shot' method for mid-circuit measurements with analytic mode."
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
    @debug_logger
    def extract_backend_info(
        device: qml.QubitDevice, capabilities: DeviceCapabilities
    ) -> BackendInfo:
        """Wrapper around extract_backend_info in the runtime module."""
        return extract_backend_info(device, capabilities)

    # pylint: disable=no-member
    @debug_logger
    def __call__(self, *args, **kwargs):
        assert isinstance(self, qml.QNode)

        # Mid-circuit measurement configuration/execution
        dynamic_one_shot_called = getattr(self, "_dynamic_one_shot_called", False)
        if not dynamic_one_shot_called:
            mcm_config = copy(self.execute_kwargs["mcm_config"])
            total_shots = get_device_shots(self.device)
            _validate_mcm_config(mcm_config, total_shots)

            if mcm_config.mcm_method == "one-shot":
                mcm_config.postselect_mode = mcm_config.postselect_mode or "hw-like"
                return dynamic_one_shot(self, mcm_config=mcm_config)(*args, **kwargs)

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
            qjit_device = QJITDevice(self.device, device_capabilities, backend_info)

        def _eval_quantum(*args):
            closed_jaxpr, out_type, out_tree = trace_quantum_function(
                self.func, qjit_device, args, kwargs, self
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


# pylint: disable=protected-access,no-member,not-callable
def dynamic_one_shot(qnode, **kwargs):
    """Transform a QNode to into several one-shot tapes to support dynamic circuit execution.

    Args:
        qnode (QNode): a quantum circuit which will run ``num_shots`` times

    Returns:
        qnode (QNode):

        The transformed circuit to be run ``num_shots`` times such as to simulate dynamic execution.


    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("lightning.qubit", shots=100)
        params = np.pi / 4 * np.ones(2)

        @qjit
        @dynamic_one_shot
        @qml.qnode(dev, diff_method=None)
        def circuit(x, y):
            qml.RX(x, wires=0)
            m0 = measure(0, reset=reset, postselect=postselect)

            @cond(m0 == 1)
            def ansatz():
                qml.RY(y, wires=1)

            ansatz()
            return measure_f(wires=[0, 1])

    The ``dynamic_one_shot`` decorator prompts the QNode to perform a hundred one-shot
    calculations, where in each calculation the ``measure`` operations dynamically
    measures the 0-wire and collapse the state vector stochastically.
    """

    cpy_tape = None
    aux_tapes = None
    mcm_config = kwargs.pop("mcm_config", None)

    def transform_to_single_shot(qnode):
        if not qnode.device.shots:
            raise qml.QuantumFunctionError("dynamic_one_shot is only supported with finite shots.")

        @qml.transform
        def dynamic_one_shot_partial(
            tape: qml.tape.QuantumTape,
        ) -> tuple[Sequence[qml.tape.QuantumTape], Callable]:
            nonlocal cpy_tape
            cpy_tape = tape
            nonlocal aux_tapes

            for m in tape.measurements:
                if not isinstance(
                    m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)
                ):
                    raise TypeError(
                        f"Native mid-circuit measurement mode does not support {type(m).__name__} "
                        "measurements."
                    )
                if isinstance(m, VarianceMP) and m.obs:
                    raise TypeError(
                        "qml.var(obs) cannot be returned when `mcm_method='one-shot'` because "
                        "the Catalyst compiler does not handle qml.sample(obs)."
                    )

            if tape.batch_size is not None:
                raise ValueError("mcm_method='one-shot' is not compatible with broadcasting")

            aux_tapes = [init_auxiliary_tape(tape)]

            def processing_fn(results):
                return results

            return aux_tapes, processing_fn

        return dynamic_one_shot_partial(qnode)

    single_shot_qnode = transform_to_single_shot(qnode)
    if mcm_config is not None:
        single_shot_qnode.execute_kwargs["mcm_config"] = mcm_config
    single_shot_qnode._dynamic_one_shot_called = True
    dev = qnode.device
    total_shots = get_device_shots(dev)

    new_dev = copy(dev)
    if isinstance(new_dev, qml.devices.LegacyDevice):
        new_dev.shots = 1  # pragma: no cover
    else:
        new_dev._shots = qml.measurements.Shots(1)
    single_shot_qnode.device = new_dev

    def one_shot_wrapper(*args, **kwargs):
        def wrap_single_shot_qnode(*_):
            return single_shot_qnode(*args, **kwargs)

        arg_vmap = jnp.empty((total_shots,), dtype=float)
        results = catalyst.vmap(wrap_single_shot_qnode)(arg_vmap)
        if isinstance(results[0], tuple) and len(results) == 1:
            results = results[0]
        return parse_native_mid_circuit_measurements(cpy_tape, aux_tapes, results, interface="jax")

    return one_shot_wrapper
