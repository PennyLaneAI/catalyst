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
"""This module contains functions tracing and lowering JAX code to MLIR.
"""

import jax
import pennylane as qml
from jax._src import source_info_util
from jax._src.dispatch import jaxpr_replicas
from jax._src.interpreters.mlir import _module_name_regex
from jax._src.lax.lax import xb, xla
from jax._src.util import wrap_name
from jax.interpreters.mlir import (
    AxisContext,
    ModuleContext,
    ReplicaAxisContext,
    ir,
    lower_jaxpr_to_fun,
    lowerable_effects,
)
from jax.interpreters.partial_eval import DynamicJaxprTracer
from jax.tree_util import tree_unflatten
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Wires

import catalyst.jax_primitives as jprim
from catalyst.jax_tape import JaxTape
from catalyst.utils.tracing import TracingContext

KNOWN_NAMED_OBS = (qml.Identity, qml.PauliX, qml.PauliY, qml.PauliZ, qml.Hadamard)


def get_mlir(func, *args, **kwargs):
    """Lower a Python function into an MLIR module.

    Args:
        func: python function to be lowered
        args: arguments to ``func``
        kwargs: keyword arguments to ``func``

    Returns:
        m: the MLIR module corresponding to ``func``
        ctx: the MLIR context corresponding
        jaxpr: the jaxpr corresponding to ``func``
    """

    # The compilation cache must be clear for each translation unit.
    # Otherwise, MLIR functions which do not exist in the current translation unit will be assumed
    # to exist if an equivalent python function is seen in the cache. This happens during testing or
    # if we wanted to compile a single python function multiple times with different options.
    jprim.mlir_fn_cache.clear()

    with TracingContext():
        jaxpr = jax.make_jaxpr(func)(*args, **kwargs)

    nrep = jaxpr_replicas(jaxpr)
    effects = [eff for eff in jaxpr.effects if eff in jax.core.ordered_effects]
    axis_context = ReplicaAxisContext(xla.AxisEnv(nrep, (), ()))
    name_stack = source_info_util.new_name_stack(wrap_name("ok", "jit"))
    module, context = custom_lower_jaxpr_to_module(
        func_name="jit_" + func.__name__,
        module_name=func.__name__,
        jaxpr=jaxpr,
        effects=effects,
        platform="cpu",
        axis_context=axis_context,
        name_stack=name_stack,
        donated_args=[],
    )

    return module, context, jaxpr


def get_traceable_fn(qfunc, device):
    """Generate a function suitable for jax tracing with custom quantum primitives.

    Args:
        qfunc: the quantum function to be traced
        device: the device in which ``qfunc`` is to be run

    Returns:
        traceable_fn: a function that when called, will trace ``qfunc``
    """

    def traceable_fn(*args, **kwargs):
        shots = device.shots
        num_wires = len(device.wires)

        jprim.qdevice("kwargs", str(device.backend_kwargs))
        jprim.qdevice("backend", device.backend_name)

        qreg = jprim.qalloc(num_wires)

        JaxTape.device = device
        with qml.QueuingManager.stop_recording():
            with JaxTape() as tape:
                with tape.quantum_tape:
                    out = qfunc(*args, **kwargs)

                return_values = out if isinstance(out, (tuple, list)) else (out,)
                meas_return_values = []
                meas_ret_val_indices = []
                non_meas_return_values = []
                for i, ret_val in enumerate(return_values):
                    if isinstance(ret_val, MeasurementProcess):
                        meas_return_values.append(ret_val)
                        meas_ret_val_indices.append(i)
                    else:
                        non_meas_return_values.append(ret_val)

                # pylint: disable=protected-access
                tape.quantum_tape._measurements = meas_return_values

                has_tracer_return_values = len(non_meas_return_values) > 0
                if has_tracer_return_values:
                    tape.set_return_val(tuple(non_meas_return_values))

                new_quantum_tape = JaxTape.device.expand_fn(tape.quantum_tape)
                tape.quantum_tape = new_quantum_tape
                tape.quantum_tape.jax_tape = tape

        return_values, _, _ = trace_quantum_tape(
            tape, qreg, has_tracer_return_values, meas_ret_val_indices, num_wires, shots
        )

        jprim.qdealloc(qreg)

        return return_values

    return traceable_fn


def insert_to_qreg(qubit_states, qreg):
    """Insert known qubit states into the quantum register.

    Args:
        qubit_states: known qubit states
        qreg: quantum register

    Returns:
        qreg: updated quantum register
    """
    for wire in qubit_states.keys():
        qreg = jprim.qinsert(qreg, wire, qubit_states[wire])
    return qreg


def get_qubits_from_wires(wires, qubit_states, qreg):
    """Get qubits corresponding to wires ``wires``.

    ``wires`` in this case may not be statically known. If at least one of the wires
    is not statically known, then it is necessary to insert all current ``qubit_states``
    in the quantum register and then query the quantum register with the statically unknown
    wire value.

    Args:
        wires: A list containing integers or ``DynamicJaxprTracer``s. The ``DynamicJaxprTracer``s
            correspond to wires which will be determined at run time.
        qubit_states: A dictionary where the keys are integers representing wires and values are
            qubit instances. ``qubit_states`` corresponds to the known assignment of each SSA qubit
            variable to a particular wire at the current program point.
        qreg: The current SSA value corresponding to the quantum register.

    Returns:
        The list of qubits queried.
    """
    has_dynamic_wire = any(map(lambda x: isinstance(x, DynamicJaxprTracer), wires))
    if has_dynamic_wire:
        qreg = insert_to_qreg(qubit_states, qreg)
    qubits = []
    for wire in wires:
        if wire not in qubit_states:
            qubits.append(jprim.qextract(qreg, wire))
        else:
            qubits.append(qubit_states[wire])
    return qubits


def get_new_qubit_state_from_wires_and_qubits(wires, new_qubits, qubit_states, qreg):
    """Update qubit state and quantum register with new qubits corresponding to wires in ``wires``.

    In the presence of any dynamic wires, it is necessary to clear the qubit states as the dynamic
    wire may have updated any previously known position.

    Args:
        wires: A list containing integers of ``DynamicJaxprTracer``s.
        new_qubits: A list corresponding to the new SSA qubit variables.
        qubit_states: The current known pairing of wires and qubits at the program point in which
                      this function is called.
        qreg: The current quantum register at the program point in which this function is called.

    Returns:
        qubit_states: An updated qubit states.
        qreg: An updated qreg.
    """
    has_dynamic_wire = any(map(lambda x: isinstance(x, DynamicJaxprTracer), wires))
    if has_dynamic_wire:
        qubit_states.clear()
    for wire, new_qubit in zip(wires, new_qubits):
        if isinstance(wire, DynamicJaxprTracer):
            qreg = jprim.qinsert(qreg, wire, new_qubit)
        else:
            qubit_states[wire] = new_qubit

    return qubit_states, qreg


# pylint: disable=too-many-statements,too-many-branches,too-many-arguments
def trace_quantum_tape(
    tape,
    qreg,
    has_tracer_return_values,
    meas_ret_val_indices=None,
    num_wires=None,
    shots=None,
):
    """Trace a quantum tape.

    Args:
        tape: the quantum tape to be traced
        qreg: the starting quantum register
        has_tracer_return_values: a boolean that indicates whether the quantum tape returns any
            values
        num_wires: the number of wires for this tape
        shots: the number of shots for this tape

    Returns:
        out: the output of the quantum tape as a ``DynamicJaxprTracer``
        qreg: the quantum register at the end of the quantum tape
        qubit_states: the qubit states at the end of the quantum tape

    """
    if meas_ret_val_indices is None:
        meas_ret_val_indices = []
    qubit_states = {}
    p = tape.get_parameter_evaluator()
    for op in tape.quantum_tape.operations:
        op_args = p.get_partial_return_value()
        if op.__class__.__name__ == "MidCircuitMeasure":
            # if mid circuit measurement, there are no parameters.
            # send the result to the ParamEvaluator
            _, wires = op_args
            assert len(wires) == 1
            qubits = get_qubits_from_wires(wires, qubit_states, qreg)
            qubit = qubits[0]
            out, new_qubit = jprim.qmeasure(qubit)
            qubit_states, qreg = get_new_qubit_state_from_wires_and_qubits(
                wires, [new_qubit], qubit_states, qreg
            )
            p.send_partial_input(out)
        elif op.__class__.__name__ == "QubitUnitary":
            matrix, wires = op_args
            qubits = get_qubits_from_wires(wires, qubit_states, qreg)
            new_qubits = jprim.qunitary(matrix, *qubits)
            qubit_states, qreg = get_new_qubit_state_from_wires_and_qubits(
                wires, new_qubits, qubit_states, qreg
            )
        elif op.__class__.__name__ == "Cond":
            predicates, consts = op_args
            qreg = insert_to_qreg(qubit_states, qreg)
            header_and_branch_args_plus_consts = predicates + consts + [qreg]
            outs = jprim.qcond(
                op.branch_jaxprs,
                *header_and_branch_args_plus_consts,
            )
            v, qreg = tree_unflatten(op.out_trees[0], outs)
            p.send_partial_input(v)
            # We don't know if the conditional modified any of the qubits
            # So let's load them all...
            qubit_states.clear()
        elif op.__class__.__name__ == "WhileLoop":
            cond_consts, body_consts, iter_args = op_args
            qreg = insert_to_qreg(qubit_states, qreg)
            iter_args_plus_consts = cond_consts + body_consts + iter_args + [qreg]
            outs = jprim.qwhile(
                op.cond_jaxpr,
                op.body_jaxpr,
                len(cond_consts),
                len(body_consts),
                *iter_args_plus_consts,
            )
            v, qreg = tree_unflatten(op.body_tree, outs)
            p.send_partial_input(v)
            # We don't know if the loop modified any of the qubits
            # So let's load them all...
            qubit_states.clear()
        elif op.__class__.__name__ == "ForLoop":
            loop_bounds, body_consts, iter_args = op_args
            qreg = insert_to_qreg(qubit_states, qreg)
            header_and_iter_args_plus_consts = loop_bounds + body_consts + iter_args + [qreg]
            outs = jprim.qfor(op.body_jaxpr, len(body_consts), *header_and_iter_args_plus_consts)
            v, qreg = tree_unflatten(op.body_tree, outs)
            p.send_partial_input(v)
            # We don't know if the loop modified any of the qubits
            # So let's load them all...
            qubit_states.clear()
        else:
            op_args, op_wires = op_args
            qubits = get_qubits_from_wires(op_wires, qubit_states, qreg)
            new_qubits = jprim.qinst(op.name, len(qubits), *qubits, *op_args)
            qubit_states, qreg = get_new_qubit_state_from_wires_and_qubits(
                op_wires, new_qubits, qubit_states, qreg
            )

    meas_return_values = []
    if len(meas_ret_val_indices) > 0:
        for meas in tape.quantum_tape.measurements:
            obs, qubits = trace_observables(meas.obs, qubit_states, p, num_wires, qreg)
            mres = trace_measurement(meas, obs, qubits, shots)
            meas_return_values.append(mres)

        assert len(meas_return_values) == len(
            meas_ret_val_indices
        ), "expected different number of measurements in qfunc output"

    return_values = []
    if has_tracer_return_values:
        ret_vals = p.get_partial_return_value()
        return_values.extend(ret_vals if isinstance(ret_vals, tuple) else (ret_vals,))

    idx_offset = 0
    for i, ret_val in zip(meas_ret_val_indices, meas_return_values):
        # Insert measurement results into the correct position of the return tuple.
        # The offset is needed for results that have more than one element, which shift
        # the position of remaining measurement results.
        idx = i + idx_offset
        return_values[idx:idx] = ret_val
        idx_offset += len(ret_val) - 1

    if len(return_values) == 1:
        out = return_values[0]
    else:
        out = tuple(return_values)

    return out, qreg, qubit_states


# TODO: remove once fixed upstream: https://github.com/PennyLaneAI/pennylane/issues/4263
def trace_hamiltonian(coeffs, *nested_obs):
    """Trace a hamiltonian.

    Args:
        coeffs: a list of coefficients
        nested_obs: a list of the nested observables

    Returns:
        a hamiltonian JAX primitive used for tracing
    """
    # jprim.hamiltonian cannot take a python list as input
    # only as *args can a list be passed as an input.
    # Instead cast it as a JAX array.
    coeffs = jax.numpy.asarray(coeffs)
    return jprim.hamiltonian(coeffs, *nested_obs)


def trace_observables(obs, qubit_states, p, num_wires, qreg):
    """Trace observables.

    Args:
        obs: an observable
        qubit_states: the statically known qubit state at this program point
        p: parameter evaluator
        num_wires: the number of wires
        qreg: the quantum register with the state at this program point

    Returns:
        jax_obs: a JAX primitive corresponding to the observable received as an argument
        qubits: a list of qubits used by the observable
    """
    op_args = p.get_partial_return_value()
    qubits = None
    if obs is None:
        _, wires = op_args
        wires = wires or Wires(range(num_wires))
        qubits = get_qubits_from_wires(wires, qubit_states, qreg)
        jax_obs = jprim.compbasis(*qubits)
    elif isinstance(obs, KNOWN_NAMED_OBS):
        _, wires = op_args
        qubits = get_qubits_from_wires(wires, qubit_states, qreg)
        jax_obs = jprim.namedobs(type(obs).__name__, qubits[0])
    elif isinstance(obs, qml.Hermitian):
        matrix, wires = op_args
        qubits = get_qubits_from_wires(wires, qubit_states, qreg)
        jax_obs = jprim.hermitian(matrix, *qubits)
    elif isinstance(obs, qml.operation.Tensor):
        nested_obs = [trace_observables(o, qubit_states, p, num_wires, qreg)[0] for o in obs.obs]
        jax_obs = jprim.tensorobs(*nested_obs)
    elif isinstance(obs, qml.Hamiltonian):
        nested_obs = [trace_observables(o, qubit_states, p, num_wires, qreg)[0] for o in obs.ops]
        jax_obs = trace_hamiltonian(op_args, *nested_obs)
    else:
        raise RuntimeError(f"unknown observable in measurement process: {obs}")
    return jax_obs, qubits


def trace_measurement(meas, obs, qubits, shots):
    """Trace measurement.

    Args:
        meas: measurement to be traced
        obs: observables used in the measurement
        qubits: qubits used in the measurement
        shots: shots used in the measurement

    Returns:
        a JAX primitive corresponding to the measurement received as an argument
    """
    compbasis = obs.primitive == jprim.compbasis_p
    if meas.return_type.value == "sample":
        shape = (shots, len(qubits)) if compbasis else (shots,)
        mres = (jprim.sample(obs, shots, shape),)
    elif meas.return_type.value == "counts":
        shape = (2 ** len(qubits),) if compbasis else (2,)
        mres = tuple(jprim.counts(obs, shots, shape))
    elif meas.return_type.value == "expval":
        mres = (jprim.expval(obs, shots),)
    elif meas.return_type.value == "var":
        mres = (jprim.var(obs, shots),)
    elif meas.return_type.value == "probs":
        assert compbasis
        shape = (2 ** len(qubits),)
        mres = (jprim.probs(obs, shape),)
    elif meas.return_type.value == "state":
        assert compbasis
        shape = (2 ** len(qubits),)
        mres = (jprim.state(obs, shape),)
    else:
        raise RuntimeError(f"unknown measurement process: {meas.return_type}")
    return mres


# pylint: disable=too-many-arguments
def custom_lower_jaxpr_to_module(
    func_name: str,
    module_name: str,
    jaxpr: jax.core.ClosedJaxpr,
    effects,
    platform: str,
    axis_context: AxisContext,
    name_stack,
    donated_args,
    replicated_args=None,
    arg_shardings=None,
    result_shardings=None,
):
    """Lowers a top-level jaxpr to an MHLO module.

    Handles the quirks of the argument/return value passing conventions of the
    runtime.

    This function has been modified from its original form in the JAX project at
    https://github.com/google/jax/blob/c4d590b1b640cc9fcfdbe91bf3fe34c47bcde917/jax/interpreters/mlir.py#L625version
    released under the Apache License, Version 2.0, with the following copyright notice:

    Copyright 2021 The JAX Authors.
    """
    platform = xb.canonicalize_platform(platform)
    if not xb.is_known_platform(platform):
        raise ValueError(f"Unknown platform {platform}")
    in_avals = jaxpr.in_avals
    assert arg_shardings is None
    assert result_shardings is None
    platforms_with_donation = ("cuda", "rocm", "tpu")
    assert platform not in platforms_with_donation
    if any(eff not in lowerable_effects for eff in jaxpr.effects):
        raise ValueError(f"Cannot lower jaxpr with effects: {jaxpr.effects}")
    if any(donated_args):
        unused_donations = [str(a) for a, d in zip(in_avals, donated_args) if d]
        msg = "See an explanation at https://jax.readthedocs.io/en/latest/faq.html#buffer-donation."
        if platform not in platforms_with_donation:
            msg = f"Donation is not implemented for {platform}.\n{msg}"

    # MHLO channels need to start at 1
    channel_iter = 1
    # Create a keepalives list that will be mutated during the lowering.
    keepalives = []
    host_callbacks = []
    ctx = ModuleContext(
        None, platform, axis_context, name_stack, keepalives, channel_iter, host_callbacks
    )
    ctx.context.allow_unregistered_dialects = True
    with ctx.context, ir.Location.unknown(ctx.context):
        # register_dialect()
        # Remove module name characters that XLA would alter. This ensures that
        # XLA computation preserves the module name.
        module_name = _module_name_regex.sub("_", module_name)
        ctx.module.operation.attributes["sym_name"] = ir.StringAttr.get(module_name)
        unlowerable_effects = {eff for eff in jaxpr.effects if eff not in lowerable_effects}
        if unlowerable_effects:
            raise ValueError(f"Cannot lower jaxpr with unlowerable effects: {unlowerable_effects}")
        lower_jaxpr_to_fun(
            ctx,
            func_name,
            jaxpr,
            effects,
            public=True,
            create_tokens=True,
            replace_tokens_with_dummy=True,
            replicated_args=replicated_args,
            arg_shardings=arg_shardings,
            result_shardings=result_shardings,
        )

        for op in ctx.module.body.operations:
            func_name = str(op.name)
            is_entry_point = func_name.startswith('"jit_')
            if is_entry_point:
                continue
            op.attributes["llvm.linkage"] = ir.Attribute.parse("#llvm.linkage<internal>")

    return ctx.module, ctx.context
