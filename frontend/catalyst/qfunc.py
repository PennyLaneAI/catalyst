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
from dataclasses import replace
from typing import Callable, Sequence

import jax.numpy as jnp
import pennylane as qml
from jax.core import eval_jaxpr
from jax.tree_util import tree_flatten, tree_unflatten
from pennylane import exceptions
from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP
from pennylane.transforms.dynamic_one_shot import (
    gather_non_mcm,
    init_auxiliary_tape,
    parse_native_mid_circuit_measurements,
)

import catalyst
from catalyst.api_extensions import MidCircuitMeasure
from catalyst.device import QJITDevice
from catalyst.jax_extras import deduce_avals, get_implicit_and_explicit_flat_args, unzip2
from catalyst.jax_primitives import quantum_kernel_p
from catalyst.jax_tracer import Function, trace_quantum_function
from catalyst.logging import debug_logger
from catalyst.passes.pass_api import dictionary_to_list_of_passes
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import filter_static_args
from catalyst.utils.exceptions import CompileError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _resolve_mcm_config(mcm_config, shots):
    """Helper function for resolving and validating that the mcm_config is valid for executing."""
    updated_values = {}

    updated_values["postselect_mode"] = (
        None if isinstance(shots, int) and shots == 0 else mcm_config.postselect_mode
    )
    if mcm_config.mcm_method is None:
        updated_values["mcm_method"] = (
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
    if mcm_config.mcm_method == "one-shot" and shots == 0:
        raise ValueError(
            "Cannot use the 'one-shot' method for mid-circuit measurements with analytic mode."
        )

    return replace(mcm_config, **updated_values)


def _get_total_shots(qnode):
    """
    Extract total shots from qnode.
    If shots is None on the qnode, this method returns 0 (static).
    This method allows the qnode shots to be either static (python int
    literals) or dynamic (tracers).
    """
    # due to possibility of tracer, we cannot use a simple `or` here to simplify
    shots_value = qnode._shots.total_shots  # pylint: disable=protected-access
    if shots_value is None:
        shots = 0
    else:
        shots = shots_value
    return shots


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
        pass_pipeline = kwargs.pop("pass_pipeline", [])
        pass_pipeline = dictionary_to_list_of_passes(pass_pipeline)

        # Mid-circuit measurement configuration/execution
        dynamic_one_shot_called = getattr(self, "_dynamic_one_shot_called", False)
        if not dynamic_one_shot_called:
            mcm_config = copy(
                qml.devices.MCMConfig(
                    postselect_mode=self.execute_kwargs["postselect_mode"],
                    mcm_method=self.execute_kwargs["mcm_method"],
                )
            )
            total_shots = _get_total_shots(self)
            mcm_config = _resolve_mcm_config(mcm_config, total_shots)

            if mcm_config.mcm_method == "one-shot":
                mcm_config = replace(
                    mcm_config, postselect_mode=mcm_config.postselect_mode or "hw-like"
                )
                return Function(dynamic_one_shot(self, mcm_config=mcm_config))(*args, **kwargs)

        new_device = copy(self.device)
        qjit_device = QJITDevice(new_device)

        static_argnums = kwargs.pop("static_argnums", ())
        out_tree_expected = kwargs.pop("_out_tree_expected", [])
        debug_info = kwargs.pop("debug_info", None)

        def _eval_quantum(*args, **kwargs):
            closed_jaxpr, out_type, out_tree, out_tree_exp = trace_quantum_function(
                self.func,
                qjit_device,
                args,
                kwargs,
                self,
                static_argnums,
                debug_info,
            )

            out_tree_expected.append(out_tree_exp)
            dynamic_args = filter_static_args(args, static_argnums)
            args_expanded = get_implicit_and_explicit_flat_args(None, *dynamic_args, **kwargs)
            res_expanded = eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.consts, *args_expanded)
            _, out_keep = unzip2(out_type)
            res_flat = [r for r, k in zip(res_expanded, out_keep) if k]
            return tree_unflatten(out_tree, res_flat)

        flattened_fun, _, _, out_tree_promise = deduce_avals(
            _eval_quantum, args, kwargs, static_argnums, debug_info
        )
        dynamic_args = filter_static_args(args, static_argnums)
        args_flat = tree_flatten((dynamic_args, kwargs))[0]
        res_flat = quantum_kernel_p.bind(
            flattened_fun, *args_flat, qnode=self, pipeline=tuple(pass_pipeline)
        )
        return tree_unflatten(out_tree_promise(), res_flat)[0]


# pylint: disable=protected-access
def _get_shot_vector(qnode):
    shot_vector = qnode._shots.shot_vector if qnode._shots else []
    return (
        shot_vector
        if len(shot_vector) > 1 or any(copies > 1 for _, copies in shot_vector)
        else None
    )


def _get_snapshot_results(tape, out):
    """
    Get the snapshot results from the tape.
    Args:
        tape: The tape to get the snapshot results from.
        out: The output of the tape.
    Returns:
        processed_snapshots: The extracted snapshot results if available;
                                otherwise, returns the original output.
        measurement_results: The corresponding measurement results.
    """
    # if no snapshot are present, return None, out
    if not any(isinstance(op, qml.Snapshot) for op in tape.operations):
        return None, out

    # Snapshots present: out[0] = snapshots, out[1] = measurements
    assert len(out) == 2
    snapshot_results, measurement_results = out

    # Take first shot for each snapshot
    processed_snapshots = [
        snapshot[0] if hasattr(snapshot, "shape") and len(snapshot.shape) > 1 else snapshot
        for snapshot in snapshot_results
    ]

    return processed_snapshots, measurement_results


def _reshape_for_shot_vector(result, shot_vector):
    # Calculate the shape for reshaping based on shot vector
    result_list = []
    start_idx = 0
    for shot, copies in shot_vector:
        # Reshape this segment to (copies, shot, n_wires)
        segment = result[start_idx : start_idx + shot * copies]
        if copies > 1:
            segment_shape = (copies, shot, result.shape[-1])
            segment = jnp.reshape(segment, segment_shape)
            result_list.extend([segment[i] for i in range(copies)])
        else:
            result_list.append(segment)
        start_idx += shot * copies
    result = tuple(result_list)
    return result


def _process_counts_measurement(out, idx, has_snapshots):
    """Process CountsMP measurement and return the result and updated index."""
    if isinstance(out[idx], tuple) and len(out[idx]) == 2:
        # CountsMP result is stored as (keys, counts) tuple
        keys, counts = out[idx]
        idx += 1
    else:
        keys = out[idx]
        counts = out[idx + 1]
        idx += 2

    if has_snapshots:
        counts_array = jnp.stack(counts, axis=0)
        aggregated_counts = jnp.sum(counts_array, axis=0)
        counts_result = (keys, aggregated_counts)
    else:
        aggregated_counts = jnp.sum(counts, axis=0)
        counts_result = (keys[0], aggregated_counts)

    return counts_result, idx


def _process_regular_measurement(m, out, idx, shot_vector):
    """Process measurements and return the result."""
    result = jnp.squeeze(out[idx])
    max_ndim = min(len(out[idx].shape), 2)
    if result.ndim == 1 and max_ndim == 2:
        result = jnp.expand_dims(result, axis=1)

    # Without MCMs and postselection, all samples are valid for use in MP computation.
    is_valid = jnp.full((result.shape[0],), True)
    processed_result = gather_non_mcm(m, result, is_valid, postselect_mode="pad-invalid-samples")

    # Handle shot vector reshaping for SampleMP
    if isinstance(m, SampleMP) and shot_vector is not None:
        processed_result = _reshape_for_shot_vector(processed_result, shot_vector)

    return processed_result


def _process_measurements_without_mcm(cpy_tape, out, snapshots, shot_vector):
    """Process measurements when there are no mid-circuit measurements."""
    new_out = []
    idx = 0

    for m in cpy_tape.measurements:
        if isinstance(m, CountsMP):
            counts_result, idx = _process_counts_measurement(out, idx, snapshots is not None)
            new_out.append(counts_result)
            continue

        processed_result = _process_regular_measurement(m, out, idx, shot_vector)
        new_out.append(processed_result)
        idx += 1

    return (snapshots, tuple(new_out)) if snapshots else tuple(new_out)


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
        if not qnode._shots:
            raise exceptions.QuantumFunctionError(
                "dynamic_one_shot is only supported with finite shots."
            )

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
    single_shot_qnode = qml.set_shots(single_shot_qnode, shots=1)
    if mcm_config is not None:
        single_shot_qnode.execute_kwargs["postselect_mode"] = mcm_config.postselect_mode
        single_shot_qnode.execute_kwargs["mcm_method"] = mcm_config.mcm_method
    single_shot_qnode._dynamic_one_shot_called = True
    total_shots = _get_total_shots(qnode)

    def one_shot_wrapper(*args, **kwargs):
        def wrap_single_shot_qnode(*_):
            return single_shot_qnode(*args, **kwargs)

        arg_vmap = jnp.empty((total_shots,), dtype=float)
        results = catalyst.vmap(wrap_single_shot_qnode)(arg_vmap)
        if isinstance(results[0], tuple) and len(results) == 1:
            results = results[0]
        has_mcm = any(isinstance(op, MidCircuitMeasure) for op in cpy_tape.operations)

        out = list(results)

        shot_vector = _get_shot_vector(qnode)
        snapshots, out = _get_snapshot_results(cpy_tape, out)

        if has_mcm and len(cpy_tape.measurements) > 0:
            out = parse_native_mid_circuit_measurements(
                cpy_tape, results=results, postselect_mode="pad-invalid-samples"
            )
            if len(cpy_tape.measurements) == 1:
                out = (out,)
        elif len(cpy_tape.measurements) > 0:
            out = _process_measurements_without_mcm(cpy_tape, out, snapshots, shot_vector)

        out_tree_expected = kwargs.pop("_out_tree_expected", [])
        if snapshots is not None:
            out = (out[0], tree_unflatten(out_tree_expected[1], out[1]))
        else:
            out = tree_unflatten(out_tree_expected[0], out)
        return out

    return one_shot_wrapper
