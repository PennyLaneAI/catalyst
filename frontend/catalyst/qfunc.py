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
from dataclasses import dataclass
from numbers import Integral
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
from catalyst.device.qjit_device import is_dynamic_wires
from catalyst.jax_extras import deduce_avals, get_implicit_and_explicit_flat_args, unzip2
from catalyst.jax_extras.tracing import uses_transform
from catalyst.jax_primitives import quantum_kernel_p
from catalyst.jax_tracer import Function, trace_quantum_function
from catalyst.logging import debug_logger
from catalyst.passes.pass_api import dict_to_compile_pipeline
from catalyst.tracing.contexts import EvaluationContext
from catalyst.tracing.type_signatures import filter_static_args
from catalyst.utils.exceptions import CompileError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class OutputContext:
    """Context containing parameters needed for finalizing quantum function output."""

    cpy_tape: any
    classical_values: any
    classical_return_indices: any
    out_tree_expected: any
    snapshots: any
    shot_vector: any
    num_mcm: int


class FallbackToSingleBranchError(Exception):
    """Exception to signal execution can proceed in single-branch-statistics mode."""


def _resolve_mcm_config(qnode):
    """Helper function for resolving and validating that the mcm_config is valid for executing."""
    total_shots = qnode._shots.total_shots  # pylint: disable=protected-access
    analytic_shots = total_shots is None
    finite_shots = not analytic_shots
    static_shots = finite_shots and isinstance(total_shots, Integral)
    device_compatible = _is_one_shot_compatible_device(qnode)

    mcm_method = qnode.execute_kwargs["mcm_method"]
    # discard the parameter when in analytic mode since it's not relevant
    postselect_mode = None if analytic_shots else qnode.execute_kwargs["postselect_mode"]

    # Check for some invalid user-configurations
    if mcm_method == "deferred":
        raise ValueError("mcm_method='deferred' is not supported with Catalyst.")

    if mcm_method == "single-branch-statistics" and postselect_mode == "hw-like":
        raise ValueError("'hw-like' post-selection requires mcm_method='one-shot'.")

    if mcm_method == "one-shot" and analytic_shots:
        raise ValueError("mcm_method='one-shot' is not supported in analytic shot mode.")

    if mcm_method == "one-shot" and not device_compatible:
        raise ValueError("mcm_method='one-shot' is not supported in the chosen device.")

    # Set the default method if none was chosen. Currently, only a static shot value
    # is supported with one-shot. Some devices also won't support it.
    if mcm_method is None:
        if static_shots and device_compatible:
            mcm_method = "one-shot"
            postselect_mode = postselect_mode or "hw-like"
        else:
            mcm_method = "single-branch-statistics"

    return qml.devices.MCMConfig(postselect_mode=postselect_mode, mcm_method=mcm_method)


def _is_one_shot_compatible_device(qnode):
    # TODO: remove the need for a hardcoded list of devices
    device_name = qnode.device.name
    exclude_devices = {"softwareq.qpp", "nvidia.custatevec", "nvidia.cutensornet"}

    # Check device name against exclude list
    if device_name in exclude_devices:
        return False

    # Additional check for OQDDevice class
    device_class_name = qnode.device.__class__.__name__
    return device_class_name != "OQDDevice"


def configure_mcm_and_try_one_shot(qnode, args, kwargs, pass_pipeline=None):
    """Configure mid-circuit measurement settings and handle one-shot execution."""
    # If we are already running a one-shot transformed QNode, do not apply MCM processing again.
    dynamic_one_shot_called = getattr(qnode, "_dynamic_one_shot_called", False)
    if dynamic_one_shot_called:
        return None

    mcm_config = _resolve_mcm_config(qnode)

    if mcm_config.mcm_method != "one-shot":
        return None

    # Special case for measurements_from_{samples/counts} transforms. If the default is determined
    # to be one-shot, but no mcms are being used, we don't want to raise an error outright. Since
    # the mcm determination happens later, we currently always fallback to single-branch if the
    # user hasn't specified one-shot explicitly. TODO: remove need for special case
    no_user_mcm_method = qnode.execute_kwargs["mcm_method"] is None
    uses_measurements_from_samples = uses_transform(qnode, "measurements_from_samples")
    uses_measurements_from_counts = uses_transform(qnode, "measurements_from_counts")
    if no_user_mcm_method and (uses_measurements_from_samples or uses_measurements_from_counts):
        return None

    # These transforms are currently not compatible with one-shot when used manually,
    # although they can be used from within a device transform program.
    if uses_measurements_from_samples:
        raise CompileError(
            "The 'measurements_from_samples' transform is not supported with the chosen or default "
            "mcm_method 'one-shot'. Please consider using 'single-branch-statistics' if you need "
            "to use this transform explicitly."
        )
    if uses_measurements_from_counts:
        raise CompileError(
            "The 'measurements_from_counts' transform is not supported with the chosen or default "
            "mcm_method 'one-shot'. Please consider using 'single-branch-statistics' if you need "
            "to use this transform explicitly."
        )

    try:
        return Function(
            dynamic_one_shot(qnode, mcm_config=mcm_config, pass_pipeline=pass_pipeline)
        )(*args, **kwargs)
    except FallbackToSingleBranchError:
        return None


def _reconstruct_output_with_classical_values(
    measurement_results, classical_values, classical_return_indices
):
    """
    Reconstruct the output values from the classical values and measurement results.
    Args:
        out: Output from measurement processing
        classical_values: Classical values
        classical_return_indices: Indices of classical values
    Returns:
        results: Reconstructed output with classical values inserted
    """
    if not classical_values:
        return measurement_results

    total_expected = len(classical_values) + len(measurement_results)
    classical_iter = iter(classical_values)
    measurement_iter = iter(measurement_results)

    def get_next_value(idx):
        return next(classical_iter) if idx in classical_return_indices else next(measurement_iter)

    results = [get_next_value(i) for i in range(total_expected)]
    return results


def _extract_classical_and_measurement_results(results, classical_return_indices):
    """
    Split results into classical values and measurement results.
    It assume that the results are in the order of classical values and measurement results.
    """
    num_classical_return_indices = len(classical_return_indices)
    classical_values = results[:num_classical_return_indices]
    measurement_results = results[num_classical_return_indices:]
    return classical_values, measurement_results


def _finalize_output(out, ctx: OutputContext):
    """
    Finalize the output by reconstructing with classical values and unflattening to the
    expected tree structure.
    Args:
        out: The output to finalize
        context: OutputContext containing all necessary parameters for finalization
    """
    # Handle case with no measurements
    if len(ctx.cpy_tape.measurements) == 0:
        out = out[: -ctx.num_mcm]

    out = _reconstruct_output_with_classical_values(
        out, ctx.classical_values, ctx.classical_return_indices
    )

    out_tree_expected = ctx.out_tree_expected
    if ctx.snapshots is not None:
        out = (out[0], tree_unflatten(out_tree_expected[1], out[1]))
    else:
        out = tree_unflatten(out_tree_expected[0], out)
    return out


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

        new_compile_pipeline, new_pass_pipeline = _extract_passes(self.compile_pipeline)
        # Update the qnode with peephole pipeline
        old_pass_pipeline = kwargs.pop("pass_pipeline", None)
        processed_old_pass_pipeline = tuple(dict_to_compile_pipeline(old_pass_pipeline))
        # Local pass pipelines
        pass_pipeline = new_pass_pipeline if new_pass_pipeline else processed_old_pass_pipeline
        new_qnode = copy(self)
        # pylint: disable=attribute-defined-outside-init, protected-access
        new_qnode._compile_pipeline = new_compile_pipeline

        # Mid-circuit measurement configuration:
        one_shot_results = configure_mcm_and_try_one_shot(new_qnode, args, kwargs, pass_pipeline)
        # If we ran one-shot execution, we are done. Otherwise, continue execution as normal.
        if one_shot_results is not None:
            return one_shot_results

        new_device = copy(new_qnode.device)
        qjit_device = QJITDevice(new_device)

        static_argnums = kwargs.pop("static_argnums", ())
        out_tree_expected = kwargs.pop("_out_tree_expected", [])
        classical_return_indices = kwargs.pop("_classical_return_indices", [])
        num_mcm_expected = kwargs.pop("_num_mcm_expected", [])
        debug_info = kwargs.pop("debug_info", None)

        def _eval_quantum(*args, **kwargs):
            trace_result = trace_quantum_function(
                new_qnode.func,
                qjit_device,
                args,
                kwargs,
                new_qnode,
                static_argnums,
                debug_info,
            )
            closed_jaxpr = trace_result.closed_jaxpr
            out_type = trace_result.out_type
            out_tree = trace_result.out_tree
            out_tree_exp = trace_result.return_values_tree
            cls_ret_idx = trace_result.classical_return_indices
            num_mcm = trace_result.num_mcm

            out_tree_expected.append(out_tree_exp)
            classical_return_indices.append(cls_ret_idx)
            num_mcm_expected.append(num_mcm)
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


def _get_snapshot_results(mcm_method, tape, out):
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
    assert mcm_method == "one-shot"

    if not any(isinstance(op, qml.Snapshot) for op in tape.operations):
        return None, out

    # Snapshots present: out[0] = snapshots, out[1] = measurements
    snapshot_results, measurement_results = out

    # Take first shot for each snapshot
    processed_snapshots = [
        snapshot[0] if hasattr(snapshot, "shape") and len(snapshot.shape) > 1 else snapshot
        for snapshot in snapshot_results
    ]

    return processed_snapshots, measurement_results


def _reshape_for_shot_vector(mcm_method, result, shot_vector):
    assert mcm_method == "one-shot"

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


def _process_terminal_measurements(mcm_method, cpy_tape, out, snapshots, shot_vector):
    """Process measurements when there are no mid-circuit measurements."""
    assert mcm_method == "one-shot"

    # flatten the outs structure
    out, _ = tree_flatten(out)
    new_out = []
    idx = 0

    for m in cpy_tape.measurements:
        if isinstance(m, CountsMP):
            if isinstance(out[idx], tuple) and len(out[idx]) == 2:
                # CountsMP result is stored as (keys, counts) tuple
                keys, counts = out[idx]
                idx += 1
            else:
                keys = out[idx]
                counts = out[idx + 1]
                idx += 2

            if snapshots is not None:
                counts_array = jnp.stack(counts, axis=0)
                aggregated_counts = jnp.sum(counts_array, axis=0)
                counts_result = (keys, aggregated_counts)
            else:
                aggregated_counts = jnp.sum(counts, axis=0)
                counts_result = (keys[0], aggregated_counts)

            new_out.append(counts_result)
            continue

        result = jnp.squeeze(out[idx])
        max_ndim = min(len(out[idx].shape), 2)
        if out[idx].shape[0] == 1:
            # Adding the first axis back when the first axis in the original
            # array is 1, since it corresponds to the shot's dimension.
            result = jnp.expand_dims(result, axis=0)
        if result.ndim == 1 and max_ndim == 2:
            result = jnp.expand_dims(result, axis=1)

        # Without MCMs and postselection, all samples are valid for use in MP computation.
        is_valid = jnp.full((result.shape[0],), True)
        processed_result = gather_non_mcm(
            m, result, is_valid, postselect_mode="pad-invalid-samples"
        )

        # Handle shot vector reshaping for SampleMP
        if isinstance(m, SampleMP) and shot_vector is not None:
            processed_result = _reshape_for_shot_vector(mcm_method, processed_result, shot_vector)

        new_out.append(processed_result)
        idx += 1

    return (snapshots, tuple(new_out)) if snapshots else tuple(new_out)


def _validate_one_shot_measurements(
    mcm_config, tape: qml.tape.QuantumTape, user_specified_mcm_method, shot_vector, wires
) -> None:
    """Validate measurements for one-shot mode.

    Args:
        mcm_config: The mid-circuit measurement configuration
        tape: The quantum tape containing measurements to validate
        qnode: The quantum node being transformed

    Raises:
        FallbackToSingleBranchError: no MCMs and the user hasn't explicitly selected one-shot
        NotImplementedError: If measurement configuration is not supported
    """
    mcm_method = mcm_config.mcm_method
    assert mcm_method == "one-shot"

    # Check if using shot vector with non-SampleMP measurements
    has_shot_vector = len(shot_vector) > 1 or any(copies > 1 for _, copies in shot_vector)
    device_has_static_wires = wires is not None and not is_dynamic_wires(wires)

    # Raise an error if there are no mid-circuit measurements, it will fallback to
    # single-branch-statistics
    if (
        not any(isinstance(op, MidCircuitMeasure) for op in tape.operations)
        and user_specified_mcm_method is None
    ):
        raise FallbackToSingleBranchError("No mid-circuit measurements present.")

    for m in tape.measurements:
        # Check one-shot restrictions based on the measurement process type.
        if not isinstance(m, (CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP)):
            raise NotImplementedError(
                f"The {type(m).__name__} measurement process is not compatible with the chosen or "
                "default mcm_method 'one-shot'. Please consider using 'single-branch-statistics' "
                "or a different measurement process."
            )

        if isinstance(m, VarianceMP) and m.obs:
            raise NotImplementedError(
                "qml.var() cannot be used on observables (MCMs are allowed) with the chosen or "
                "default mcm_method 'one-shot'. Please consider using 'single-branch-statistics' "
                "or using it on MCMs instead."
            )

        if has_shot_vector and not isinstance(m, SampleMP):
            raise NotImplementedError(
                f"The {type(m).__name__} measurement process does not support shot-vectors "
                "with the chosen or default mcm_method 'one-shot'. Please consider using "
                "'single-branch-statistics' or qml.sample() instead."
            )

        if (
            isinstance(m, (SampleMP, CountsMP))
            and not m.wires.tolist()
            and not device_has_static_wires
        ):
            raise NotImplementedError(
                "qml.sample() and qml.counts() cannot be used without wires and a dynamic number "
                "of device wires in the chosen or default mcm_method 'one-shot'. Please consider "
                "using 'single-branch-statistics', providing explicit wires in the measurement "
                "process, or setting a constant number of wires in the device."
            )


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

        dev = qml.device("lightning.qubit")
        params = np.pi / 4 * np.ones(2)

        @qjit
        @dynamic_one_shot
        @qml.qnode(dev, diff_method=None, shots=100)
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
    mcm_config = kwargs.pop("mcm_config", None)
    pass_pipeline = kwargs.pop("pass_pipeline", None)

    def transform_to_single_shot(qnode):
        if not qnode._shots:
            raise exceptions.QuantumFunctionError(
                "dynamic_one_shot is only supported with finite shots."
            )

        user_specified_mcm_method = qnode.execute_kwargs["mcm_method"]
        shot_vector = qnode._shots.shot_vector if qnode._shots else []
        wires = qnode.device.wires

        @qml.transform
        def dynamic_one_shot_partial(
            tape: qml.tape.QuantumTape,
        ) -> tuple[Sequence[qml.tape.QuantumTape], Callable]:
            nonlocal cpy_tape
            cpy_tape = tape

            _validate_one_shot_measurements(
                mcm_config, tape, user_specified_mcm_method, shot_vector, wires
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
    total_shots = qnode._shots.total_shots
    assert total_shots is not None, "must have a shot value for one-shot"

    def one_shot_wrapper(*args, **kwargs):
        if pass_pipeline is not None:
            kwargs["pass_pipeline"] = pass_pipeline

        def wrap_single_shot_qnode(*_):
            return single_shot_qnode(*args, **kwargs)

        arg_vmap = jnp.empty((total_shots,), dtype=float)
        results = catalyst.vmap(wrap_single_shot_qnode)(arg_vmap)
        if isinstance(results[0], tuple) and len(results) == 1:
            results = results[0]
        has_mcm = any(isinstance(op, MidCircuitMeasure) for op in cpy_tape.operations)

        classical_return_indices = kwargs.pop("_classical_return_indices", [[]])[0]
        num_mcm = kwargs.pop("_num_mcm_expected", [0])[0]
        out_tree_expected = kwargs.pop("_out_tree_expected", [[]])

        # Split results into classical and measurement parts
        classical_values, results = _extract_classical_and_measurement_results(
            results, classical_return_indices
        )

        out = list(results)

        shot_vector = _get_shot_vector(qnode)
        snapshots, out = _get_snapshot_results(mcm_config.mcm_method, cpy_tape, out)

        if has_mcm and len(cpy_tape.measurements) > 0:
            out = parse_native_mid_circuit_measurements(
                cpy_tape, results=results, postselect_mode="pad-invalid-samples"
            )
            if len(cpy_tape.measurements) == 1:
                out = (out,)
        elif len(cpy_tape.measurements) > 0:
            out = _process_terminal_measurements(
                mcm_config.mcm_method, cpy_tape, out, snapshots, shot_vector
            )

        ctx = OutputContext(
            cpy_tape=cpy_tape,
            classical_values=classical_values,
            classical_return_indices=classical_return_indices,
            out_tree_expected=out_tree_expected,
            snapshots=snapshots,
            shot_vector=shot_vector,
            num_mcm=num_mcm,
        )

        return _finalize_output(out, ctx)

    return one_shot_wrapper


def _extract_passes(transform_program):
    """Extract transforms with pass names from the end of the CompilePipeline."""
    tape_transforms = []
    pass_pipeline = []
    i = len(transform_program)
    for t in reversed(transform_program):
        if t.pass_name is None:
            break
        i -= 1
    pass_pipeline = transform_program[i:]
    tape_transforms = transform_program[:i]
    for t in tape_transforms:
        if t.tape_transform is None:
            raise ValueError(
                f"{t} without a tape definition occurs before tape transform {tape_transforms[-1]}."
            )
    return qml.CompilePipeline(tape_transforms), tuple(pass_pipeline)
