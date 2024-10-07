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
    gather_non_mcm,
    init_auxiliary_tape,
    parse_native_mid_circuit_measurements,
)

import catalyst
from catalyst.api_extensions import MidCircuitMeasure
from catalyst.device import QJITDevice, get_device_shots
from catalyst.jax_extras import (
    deduce_avals,
    get_implicit_and_explicit_flat_args,
    unzip2,
)
from catalyst.jax_primitives import quantum_kernel_p
from catalyst.jax_tracer import trace_quantum_function, Function
from catalyst.logging import debug_logger
from catalyst.passes import pipeline
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import filter_static_args
from catalyst.utils.exceptions import CompileError

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

    # pylint: disable=no-member
    # pylint: disable=self-cls-assignment
    @debug_logger
    def __call__(self, *args, **kwargs):

        if EvaluationContext.is_quantum_tracing():
            raise CompileError("Can't nest qnodes under qjit")

        assert isinstance(self, qml.QNode)

        # Update the qnode with peephole pipeline
        if "pass_pipeline" in kwargs.keys():
            pass_pipeline = kwargs["pass_pipeline"]
            if not hasattr(self, "_peephole_transformed"):
                self = pipeline(pass_pipeline=pass_pipeline)(self)
            kwargs.pop("pass_pipeline")

        # Mid-circuit measurement configuration/execution
        dynamic_one_shot_called = getattr(self, "_dynamic_one_shot_called", False)
        if not dynamic_one_shot_called:
            mcm_config = copy(self.execute_kwargs["mcm_config"])
            total_shots = get_device_shots(self.device)
            _validate_mcm_config(mcm_config, total_shots)

            if mcm_config.mcm_method == "one-shot":
                mcm_config.postselect_mode = mcm_config.postselect_mode or "hw-like"
                return Function(dynamic_one_shot(self, mcm_config=mcm_config))(*args, **kwargs)

        qjit_device = QJITDevice(self.device)

        static_argnums = kwargs.pop("static_argnums", ())
        out_tree_expected = kwargs.pop("_out_tree_expected", [])

        def _eval_quantum(*args, **kwargs):
            closed_jaxpr, out_type, out_tree, out_tree_exp = trace_quantum_function(
                self.func,
                qjit_device,
                args,
                kwargs,
                self,
                static_argnums,
            )

            out_tree_expected.append(out_tree_exp)
            dynamic_args = filter_static_args(args, static_argnums)
            args_expanded = get_implicit_and_explicit_flat_args(None, *dynamic_args, **kwargs)
            res_expanded = eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args_expanded)
            _, out_keep = unzip2(out_type)
            res_flat = [r for r, k in zip(res_expanded, out_keep) if k]
            return tree_unflatten(out_tree, res_flat)

        flattened_fun, _, _, out_tree_promise = deduce_avals(
            _eval_quantum, args, kwargs, static_argnums
        )
        dynamic_args = filter_static_args(args, static_argnums)
        args_flat = tree_flatten((dynamic_args, kwargs))[0]
        res_flat = quantum_kernel_p.bind(flattened_fun, *args_flat, qnode=self)
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
    if isinstance(new_dev, qml.devices.LegacyDeviceFacade):
        new_dev.target_device.shots = 1  # pragma: no cover
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
        has_mcm = any(isinstance(op, MidCircuitMeasure) for op in cpy_tape.operations)
        out = list(results)
        if has_mcm:
            out = parse_native_mid_circuit_measurements(
                cpy_tape, aux_tapes, results, postselect_mode="pad-invalid-samples"
            )
            if len(cpy_tape.measurements) == 1:
                out = (out,)
        else:
            for m_count, m in enumerate(cpy_tape.measurements):
                # Without MCMs and postselection, all samples are valid for use in MP computation.
                is_valid = jnp.array([True] * len(out[m_count]))
                out[m_count] = gather_non_mcm(
                    m, out[m_count], is_valid, postselect_mode="pad-invalid-samples"
                )
            out = tuple(out)
        out_tree_expected = kwargs.pop("_out_tree_expected", [])
        out = tree_unflatten(out_tree_expected[0], out)
        return out

    return one_shot_wrapper
