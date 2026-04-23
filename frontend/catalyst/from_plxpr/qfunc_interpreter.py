# Copyright 2025 Xanadu Quantum Technologies Inc.

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
Sets up the PLxPRToQuantumJaxprInterpreter for converting plxpr to catalyst jaxpr.
"""

# pylint: disable=protected-access
import textwrap
from copy import copy
from functools import partial

import jax
import jax.numpy as jnp
import pennylane as qml
from jax._src.sharding_impls import UNSPECIFIED
from jax.interpreters.partial_eval import DynamicJaxprTracer
from pennylane.capture import PlxprInterpreter, pause
from pennylane.capture.primitives import ctrl_transform_prim as plxpr_ctrl_transform_prim
from pennylane.capture.primitives import measure_prim as plxpr_measure_prim
from pennylane.capture.primitives import pauli_measure_prim as plxpr_pauli_measure_prim
from pennylane.capture.primitives import quantum_subroutine_prim, transform_prim
from pennylane.ftqc.primitives import measure_in_basis_prim as plxpr_measure_in_basis_prim
from pennylane.measurements import CountsMP

from catalyst.from_plxpr.qref_jax_primitives import (
    QrefQubit,
    qref_alloc_p,
    qref_compbasis_p,
    qref_dealloc_p,
    qref_get_p,
    qref_gphase_p,
    qref_hermitian_p,
    qref_measure_p,
    qref_namedobs_p,
    qref_pauli_rot_p,
    qref_qinst_p,
    qref_set_basis_state_p,
    qref_set_state_p,
    qref_unitary_p,
)
from catalyst.jax_extras import jaxpr_pad_consts
from catalyst.jax_primitives import (
    AbstractQbit,
    MeasurementPlane,
    cond_p,
    counts_p,
    decomprule_p,
    expval_p,
    hamiltonian_p,
    mcmobs_p,
    measure_in_basis_p,
    pauli_measure_p,
    probs_p,
    qinst_p,
    sample_p,
    state_p,
    tensorobs_p,
    var_p,
)
from catalyst.utils.exceptions import CompileError

from .qubit_handler import (
    QubitHandler,
    QubitIndexRecorder,
    _get_dynamically_allocated_qregs,
    get_in_qubit_values,
    is_dynamically_allocated_wire,
)

measurement_map = {
    qml.measurements.SampleMP: sample_p,
    qml.measurements.ExpectationMP: expval_p,
    qml.measurements.VarianceMP: var_p,
    qml.measurements.ProbabilityMP: probs_p,
    qml.measurements.StateMP: state_p,
}


def _flat_prod_gen(op: qml.ops.Prod):
    for o in op:
        if isinstance(o, qml.ops.Prod):
            yield from _flat_prod_gen(o)
        else:
            yield o


# pylint: disable=too-many-instance-attributes
class PLxPRToQuantumJaxprInterpreter(PlxprInterpreter):
    """
    Unlike the previous interpreters which modified the getattr and setattr
    and maintained a stack of references to the quantum register to be used
    as an access path in stack allocated objects, this translator receives
    the qreg over which qubits will be taken and inserted into as a parameter
    during initialization.
    """

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        device,
        shots,
        init_qreg,
        cache,
        *,
        control_wires=(),
        control_values=(),
    ):
        self.device = device
        self.shots = shots
        self.init_qreg = init_qreg
        self.subroutine_cache = cache
        self.control_wires = control_wires
        """Any control wires used for a subroutine."""
        self.control_values = control_values
        """Any control values for executing a subroutine."""
        self.has_dynamic_allocation = False

        super().__init__()

    def interpret_operation(self, op, is_adjoint=False, control_values=(), control_wires=()):
        """Re-bind a pennylane operation as a catalyst instruction.

        Note that this method handles the unwrapping of adjoint and controlled
        operations recursively.

        For special operations that require custom parameter handling during
        binding, we refer to the `_special_op_bind_call` mapping.

        Args:
            op (qml.operation.Operator): The operation to interpret.
            is_adjoint (bool): Whether the operation is in adjoint mode.
            control_values (tuple): Control values for controlled operations.
            control_wires (tuple): Control wires for controlled operations.

        Returns:
            List[AbstractQbit]: The resulting qubits after applying the operation.
        """
        if isinstance(op, qml.ops.Adjoint):
            return self.interpret_operation(
                op.base,
                is_adjoint=not is_adjoint,
                control_values=control_values,
                control_wires=control_wires,
            )
        if type(op) in {qml.ops.Controlled, qml.ops.ControlledOp}:
            return self.interpret_operation(
                op.base,
                is_adjoint=is_adjoint,
                control_values=control_values + tuple(op.control_values),
                control_wires=control_wires + tuple(op.control_wires),
            )

        control_wires = control_wires + self.control_wires
        control_values = control_values + self.control_values

        if (fn := _special_op_bind_call.get(type(op))) is not None:
            bind_fn = partial(fn, hyperparameters=op.hyperparameters)
        else:
            bind_fn = qref_qinst_p.bind

        in_qubits = []
        in_control_qubits = []
        for w in op.wires:
            if isinstance(w, DynamicJaxprTracer) and isinstance(w.val.aval, QrefQubit):
                in_qubits.append(w)
            else:
                in_qubits.append(qref_get_p.bind(self.init_qreg, w))
        for w in control_wires:
            if isinstance(w, DynamicJaxprTracer) and isinstance(w.val.aval, QrefQubit):
                in_control_qubits.append(w)
            else:
                in_control_qubits.append(qref_get_p.bind(self.init_qreg, w))

        bind_fn(
            *[*in_qubits, *op.data, *in_control_qubits, *control_values],
            op=op.name,
            qubits_len=len(op.wires),
            params_len=len(op.data),
            ctrl_len=len(control_wires),
            adjoint=is_adjoint,
        )

        return ()

    def _obs(self, obs):
        """Interpret the observable equation corresponding to a measurement equation's input."""
        if isinstance(obs, qml.ops.Prod):
            # catalyst cant handle product of products
            return tensorobs_p.bind(*(self._obs(t) for t in _flat_prod_gen(obs)))
        if obs.arithmetic_depth > 0:
            with pause():
                coeffs, terms = obs.terms()
            terms = [self._obs(t) for t in terms]
            return hamiltonian_p.bind(jnp.stack(coeffs), *terms)
        wires = [qref_get_p.bind(self.init_qreg, w) for w in obs.wires]
        if obs.name == "Hermitian":
            return qref_hermitian_p.bind(obs.data[0], *wires)
        return qref_namedobs_p.bind(wires[0], kind=obs.name)

    def _compbasis_obs(self, *wires):
        """Add a computational basis sampling observable."""
        if wires:
            qubits = [qref_get_p.bind(self.init_qreg, w) for w in wires]
            return qref_compbasis_p.bind(*qubits)
        else:
            return qref_compbasis_p.bind(self.init_qreg, qreg_available=True)

    def _check_measurement_with_dynamic_allocation(self, measurement):
        """Check some constraints regarding dynamic allocation."""
        if self.has_dynamic_allocation:
            if len(measurement.wires) == 0 and not isinstance(
                measurement, qml.measurements.StateMP
            ):
                raise CompileError(textwrap.dedent("""
                        Terminal measurements must take in an explicit list of wires when
                        dynamically allocated wires are present in the program.
                        """))

            if any(is_dynamically_allocated_wire(w) for w in measurement.wires):
                raise CompileError(textwrap.dedent("""
                        Terminal measurements cannot take in dynamically allocated wires
                        since they must be temporary.
                        """))

    # pylint: disable=too-many-branches
    def interpret_measurement(self, measurement):
        """Rebind a measurement as a catalyst instruction.

        Args:
            measurement (qml.measurements.MeasurementProcess): The measurement to interpret.

        Returns:
            AbstractQbit: The resulting measurement value.
        """
        self._check_measurement_with_dynamic_allocation(measurement)

        if type(measurement) not in measurement_map:
            raise NotImplementedError(
                f"measurement {measurement} not yet supported for conversion."
            )

        if measurement._eigvals is not None:
            raise NotImplementedError(
                "from_plxpr does not yet support measurements with manual eigvals."
            )

        prim = measurement_map[type(measurement)]
        mcm_mv = measurement.mv
        mcm_obs = measurement.obs
        mp_is_on_mcm = (mcm_mv is not None) or (
            mcm_obs is not None and not isinstance(mcm_obs, qml.operation.Operator)
        )

        if mp_is_on_mcm:
            mcm_list = mcm_mv if isinstance(mcm_mv, list) else [mcm_mv]
            obs = mcmobs_p.bind(*mcm_list)
            n_wires = len(mcm_list)
            _abstract_eval_kwargs = {}
        else:
            obs = self._obs(mcm_obs) if mcm_obs else self._compbasis_obs(*measurement.wires)
            n_wires = len(measurement.wires)
            _abstract_eval_kwargs = {"num_device_wires": len(self.device.wires)}

        shape, dtype = measurement._abstract_eval(
            n_wires=n_wires, shots=self.shots, **_abstract_eval_kwargs
        )

        if prim is sample_p:
            if mp_is_on_mcm:
                sample_shape = (self.shots, n_wires)
            else:
                num_qubits = len(measurement.wires) or len(self.device.wires)
                sample_shape = (self.shots, num_qubits)
            dyn_dims, static_shape = jax._src.lax.lax._extract_tracers_dyn_shape(sample_shape)
            mval = sample_p.bind(obs, *dyn_dims, static_shape=tuple(static_shape))
        elif prim in {expval_p, var_p}:
            mval = prim.bind(obs, shape=shape)
        else:
            dyn_dims, static_shape = jax._src.lax.lax._extract_tracers_dyn_shape(shape)
            mval = prim.bind(obs, *dyn_dims, static_shape=tuple(static_shape))

        # sample_p returns floats, so we need to converted it back to the expected integers here
        if dtype != mval.dtype:
            return jax.lax.convert_element_type(mval, dtype)
        return mval

    def __call__(self, jaxpr, *args):
        """
        Execute this interpreter with this arguments.
        We expect this to be a flat function (i.e., always takes *args as inputs
        and no **kwargs) and the results is a sequence of values
        """
        return self.eval(jaxpr.jaxpr, jaxpr.consts, *args)


# pylint: disable=unused-argument, too-many-arguments
def _qubit_unitary_bind_call(
    *invals, op, qubits_len, params_len, ctrl_len, adjoint, hyperparameters
):
    wires = invals[:qubits_len]
    mat = invals[qubits_len]
    ctrl_inputs = invals[qubits_len + 1 :]
    return qref_unitary_p.bind(
        mat, *wires, *ctrl_inputs, qubits_len=qubits_len, ctrl_len=ctrl_len, adjoint=adjoint
    )


# pylint: disable=unused-argument, too-many-arguments
def _gphase_bind_call(*invals, op, qubits_len, params_len, ctrl_len, adjoint, hyperparameters):
    return qref_gphase_p.bind(*invals[qubits_len:], ctrl_len=ctrl_len, adjoint=adjoint)


# pylint: disable=too-many-arguments
def _pcphase_bind_call(*invals, op, qubits_len, params_len, ctrl_len, adjoint, hyperparameters):
    wires = invals[:qubits_len]
    angle = invals[qubits_len]

    # This is a temporary workaround to properly capture
    # the dimension of the subspace for the PCPhase operation
    # which does not follow the same pattern as `qinst_p`.
    # We will revisit this once we have a better solution for
    # supporting general PL operations in the capture framework.
    # See https://docs.pennylane.ai/en/stable/code/api/pennylane.PCPhase.html
    dim = hyperparameters["dimension"][0]
    params_len += 1

    ctrl_inputs = invals[qubits_len + 2 :]
    ctrl_wires = invals[qubits_len + 1 : -len(ctrl_inputs)]

    return qref_qinst_p.bind(
        *wires,
        angle,
        dim,
        *ctrl_wires,
        *ctrl_inputs,
        op=op,
        qubits_len=qubits_len,
        params_len=params_len,
        ctrl_len=ctrl_len,
        adjoint=adjoint,
    )


# pylint: disable=unused-argument, too-many-arguments
def _pauli_rot_bind_call(*invals, op, qubits_len, params_len, ctrl_len, adjoint, hyperparameters):
    """Handle the conversion from plxpr to Catalyst jaxpr for the PauliMeasure primitive"""
    # invals are the input wires
    wires = invals[:qubits_len]
    params = invals[qubits_len : qubits_len + params_len]
    pauli_word = hyperparameters["pauli_word"]
    ctrl_wires = invals[qubits_len + params_len : qubits_len + params_len + ctrl_len]
    ctrl_values = invals[qubits_len + params_len + ctrl_len :]
    return qref_pauli_rot_p.bind(
        *[*wires, *params, *ctrl_wires, *ctrl_values],
        pauli_word=pauli_word,
        qubits_len=qubits_len,
        params_len=params_len,
        ctrl_len=ctrl_len,
        adjoint=adjoint,
    )


# Mapping of special operations to their bind calls
# These operations require special handling of their parameters
# during the binding process.
_special_op_bind_call = {
    qml.QubitUnitary: _qubit_unitary_bind_call,
    qml.GlobalPhase: _gphase_bind_call,
    qml.PCPhase: _pcphase_bind_call,
    qml.PauliRot: _pauli_rot_bind_call,
}


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.allocation.allocate_prim)
def handle_qml_alloc(self, *, num_wires, state=None, restored=False):
    """Handle the conversion from plxpr to Catalyst jaxpr for the qml.allocate primitive"""

    assert isinstance(
        num_wires, int
    ), "number of dynamically allocated qubits must be statically known"

    self.has_dynamic_allocation = True
    new_qreg = qref_alloc_p.bind(static_num_qubits=num_wires)
    return [qref_get_p.bind(new_qreg, i) for i in range(num_wires)]


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.allocation.deallocate_prim)
def handle_qml_dealloc(self, *wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the qml.deallocate primitive"""
    qregs = set()
    for w in wires:
        get_op = w.parent
        assert (
            get_op.primitive is qref_get_p
        ), "Manual deallocation is only supported for manually allocated wires"
        qreg = get_op.in_tracers[0]
        qregs.add(qreg)
    assert (
        len(qregs) == 1
    ), "Expected all wires to deallocate to come from the same allocation instruction"
    qref_dealloc_p.bind(list(qregs)[0])
    return []


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(CountsMP._wires_primitive)
def interpret_counts(self, *wires, all_outcomes):
    """Interpret a CountsMP primitive as the catalyst version."""
    obs = self._compbasis_obs(*wires)
    num_wires = len(wires) if wires else len(self.device.wires)
    keys, vals = counts_p.bind(obs, static_shape=(2**num_wires,))
    keys = jax.lax.convert_element_type(keys, int)
    return keys, vals


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(CountsMP._mcm_primitive)
def interpret_counts_mcm(self, *mcms, single_mcm, all_outcomes):
    """Interpret a CountsMP MCM primitive as the catalyst version."""
    obs = mcmobs_p.bind(*mcms)
    keys, vals = counts_p.bind(obs, static_shape=(2 ** len(mcms),))
    keys = jax.lax.convert_element_type(keys, int)
    return keys, vals


def _subroutine_kernel(
    interpreter,
    jaxpr,
    *qregs_plus_args,
    outer_dynqreg_handlers=(),
    wire_label_arg_to_tracer_arg_index=(),
    wire_to_owner_qreg=(),
):
    global_qreg, *dynqregs_plus_args = qregs_plus_args
    num_dynamic_alloced_qregs = len(outer_dynqreg_handlers)
    dynalloced_qregs, args = (
        dynqregs_plus_args[:num_dynamic_alloced_qregs],
        dynqregs_plus_args[num_dynamic_alloced_qregs:],
    )

    # Launch a new interpreter for the body region
    # A new interpreter's root qreg value needs a new recorder
    converter = copy(interpreter)
    converter.qubit_index_recorder = QubitIndexRecorder()
    init_qreg = QubitHandler(global_qreg, converter.qubit_index_recorder)
    converter.init_qreg = init_qreg

    # add dynamic qregs to recorder
    qreg_map = {}
    dyn_qreg_handlers = []
    arg_to_qreg = {}
    for dyn_qreg, outer_dynqreg_handler in zip(
        dynalloced_qregs, outer_dynqreg_handlers, strict=True
    ):
        dyn_qreg_handler = QubitHandler(dyn_qreg, converter.qubit_index_recorder)
        dyn_qreg_handlers.append(dyn_qreg_handler)

        # plxpr global wire index does not change across scopes
        # So scope arg dynamic qregs need to have the same root hash as their corresponding
        # qreg tracers outside
        dyn_qreg_handler.root_hash = outer_dynqreg_handler.root_hash

        # Each qreg argument of the subscope corresponds to a qreg from the outer scope
        qreg_map[outer_dynqreg_handler] = dyn_qreg_handler

    for global_idx, arg_idx in wire_label_arg_to_tracer_arg_index.items():
        arg_to_qreg[args[arg_idx]] = qreg_map[wire_to_owner_qreg[global_idx]]

    # The new interpreter's recorder needs to be updated to include the qreg args
    # of this scope, instead of the outer qregs
    for arg in args:
        if arg in arg_to_qreg:
            converter.qubit_index_recorder[arg] = arg_to_qreg[arg]

    retvals = converter(jaxpr, *args)

    init_qreg.insert_all_dangling_qubits()

    # Return all registers
    for dyn_qreg_handler in reversed(dyn_qreg_handlers):
        dyn_qreg_handler.insert_all_dangling_qubits()
        retvals.insert(0, dyn_qreg_handler.get())

    return converter.init_qreg.get(), *retvals


@PLxPRToQuantumJaxprInterpreter.register_primitive(quantum_subroutine_prim)
def handle_subroutine(self, *args, **kwargs):
    """
    Transform the subroutine from PLxPR into JAXPR with quantum primitives.
    """

    backup = dict(self.init_qreg)
    self.init_qreg.insert_all_dangling_qubits()

    # Make sure the quantum register is updated
    plxpr = kwargs["jaxpr"]
    transformed = self.subroutine_cache.get(plxpr)

    dynalloced_qregs, dynalloced_wire_global_indices = _get_dynamically_allocated_qregs(
        args, self.qubit_index_recorder, self.init_qreg
    )
    wire_to_owner_qreg = dict(zip(dynalloced_wire_global_indices, dynalloced_qregs))
    dynalloced_qregs = list(dict.fromkeys(dynalloced_qregs))  # squash duplicates

    # Convert global wire indices into local indices
    new_args = ()
    wire_label_arg_to_tracer_arg_index = {}
    for i, arg in enumerate(args):
        if arg in dynalloced_wire_global_indices:
            wire_label_arg_to_tracer_arg_index[arg] = i
            new_args += (self.qubit_index_recorder[arg].global_index_to_local_index(arg),)
        else:
            new_args += (arg,)

    if not transformed:
        f = partial(
            _subroutine_kernel,
            self,
            plxpr,
            outer_dynqreg_handlers=dynalloced_qregs,
            wire_label_arg_to_tracer_arg_index=wire_label_arg_to_tracer_arg_index,
            wire_to_owner_qreg=wire_to_owner_qreg,
        )
        converted_closed_jaxpr_branch = jax.make_jaxpr(f)(
            self.init_qreg.get(), *[dyn_qreg.get() for dyn_qreg in dynalloced_qregs], *args
        )
        self.subroutine_cache[plxpr] = converted_closed_jaxpr_branch
    else:
        converted_closed_jaxpr_branch = transformed

    # quantum_subroutine_p.bind
    # is just pjit_p with a different name.
    vals_out = quantum_subroutine_prim.bind(
        self.init_qreg.get(),
        *[dyn_qreg.get() for dyn_qreg in dynalloced_qregs],
        *new_args,
        jaxpr=converted_closed_jaxpr_branch,
        in_shardings=(*(UNSPECIFIED,) * (len(dynalloced_qregs) + 1), *kwargs["in_shardings"]),
        out_shardings=(*(UNSPECIFIED,) * (len(dynalloced_qregs) + 1), *kwargs["out_shardings"]),
        in_layouts=(*(None,) * (len(dynalloced_qregs) + 1), *kwargs["in_layouts"]),
        out_layouts=(*(None,) * (len(dynalloced_qregs) + 1), *kwargs["out_layouts"]),
        donated_invars=kwargs["donated_invars"],
        ctx_mesh=kwargs["ctx_mesh"],
        name=kwargs["name"],
        keep_unused=kwargs["keep_unused"],
        inline=kwargs["inline"],
        compiler_options_kvs=kwargs["compiler_options_kvs"],
    )

    self.init_qreg.set(vals_out[0])
    for i, dyn_qreg in enumerate(dynalloced_qregs):
        dyn_qreg.set(vals_out[i + 1])
    vals_out = vals_out[len(dynalloced_qregs) + 1 :]

    for orig_wire in backup.keys():
        self.init_qreg.extract(orig_wire)

    return vals_out


@PLxPRToQuantumJaxprInterpreter.register_primitive(decomprule_p)
def handle_decomposition_rule(self, *, pyfun, func_jaxpr, is_qreg, num_params):
    """
    Transform a quantum decomposition rule from PLxPR into JAXPR with quantum primitives.
    """
    if is_qreg:
        self.init_qreg.insert_all_dangling_qubits()

        def wrapper(qreg, *args):
            # Launch a new interpreter for the new subroutine region
            # A new interpreter's root qreg value needs a new recorder
            converter = copy(self)
            converter.qubit_index_recorder = QubitIndexRecorder()
            init_qreg = QubitHandler(qreg, converter.qubit_index_recorder)
            converter.init_qreg = init_qreg

            converter(func_jaxpr, *args)
            converter.init_qreg.insert_all_dangling_qubits()
            return converter.init_qreg.get()

        converted_closed_jaxpr_branch = jax.make_jaxpr(wrapper)(
            self.init_qreg.get(), *func_jaxpr.in_avals
        )
    else:

        def wrapper(*args):
            # Launch a new interpreter for the new subroutine region
            # A new interpreter's root qreg value needs a new recorder

            # TODO: it is a bit messy that the qubit mode of decompositions,
            # which just needs to keep track of a list of explicit qubit's latest SSA values,
            # is going through the entire qreg value mapping infra.
            # Two bitter things here are that:
            #   - qubit lists do not need a recorder (they don't need to remember which qubits
            #     belong to which qregs)
            #   - the qubit list object needs to piggy-back off the `init_qreg` attribute of the
            #     interpreter, which is a wrong name for this case
            # We should refactor the QubitHandler object into a qubit mode object and a qreg
            # mode object.

            converter = copy(self)
            qubit_handler = QubitHandler(args[num_params:], recorder=None)
            converter.init_qreg = qubit_handler

            converter(func_jaxpr, *args)
            return converter.init_qreg.get()

        new_in_avals = func_jaxpr.in_avals[:num_params] + [
            AbstractQbit() for _ in func_jaxpr.in_avals[num_params:]
        ]
        converted_closed_jaxpr_branch = jax.make_jaxpr(wrapper)(*new_in_avals)

    decomprule_p.bind(pyfun=pyfun, func_jaxpr=converted_closed_jaxpr_branch)

    return ()


@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_pauli_measure_prim)
def handle_pauli_measure(self, *invals, pauli_word, **params):
    """Handle the conversion from plxpr to Catalyst jaxpr for the PauliMeasure primitive"""
    # invals are the input wires
    in_qregs, in_qubits = get_in_qubit_values(invals, self.qubit_index_recorder, self.init_qreg)
    outvals = pauli_measure_p.bind(*in_qubits, pauli_word=pauli_word, qubits_len=len(in_qubits))
    result, *out_qubits = outvals  # First element is the measurement result
    for in_qreg, w, new_wire in zip(in_qregs, invals, out_qubits):
        in_qreg[in_qreg.global_index_to_local_index(w)] = new_wire
    result = jnp.astype(result, int)
    return result


@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.BasisState._primitive)
def handle_basis_state(self, *invals, n_wires):
    """Handle the conversion from plxpr to Catalyst jaxpr for the BasisState primitive"""
    state_inval = invals[0]
    wires_inval = invals[1:]
    in_qubits = []
    for w in wires_inval:
        if isinstance(w, DynamicJaxprTracer) and isinstance(w.val.aval, QrefQubit):
            in_qubits.append(w)
        else:
            in_qubits.append(qref_get_p.bind(self.init_qreg, w))

    state = jax.lax.convert_element_type(state_inval, jnp.dtype(jnp.bool))

    qref_set_basis_state_p.bind(*in_qubits, state)


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(qml.StatePrep._primitive)
def handle_state_prep(self, *invals, n_wires, **kwargs):
    """Handle the conversion from plxpr to Catalyst jaxpr for the StatePrep primitive"""
    state_inval = invals[0]
    wires_inval = invals[1:]
    in_qubits = []
    for w in wires_inval:
        if isinstance(w, DynamicJaxprTracer) and isinstance(w.val.aval, QrefQubit):
            in_qubits.append(w)
        else:
            in_qubits.append(qref_get_p.bind(self.init_qreg, w))

    normalize = kwargs.get("normalize", False)
    pad_with = kwargs.get("pad_with")
    if pad_with is not None:
        normalize = True
        n_states = state_inval.shape[-1]
        dim = 2**n_wires
        if n_states < dim:
            pad_with = jnp.array(pad_with, dtype=state_inval.dtype)
            state_inval = jax.lax.pad(state_inval, pad_with, [(0, dim - n_states, 0)])

    if normalize:
        norm = jnp.linalg.norm(state_inval)
        state_inval /= norm

    # jnp.complex128 is the top element in the type promotion lattice so it is ok to do this:
    # https://jax.readthedocs.io/en/latest/type_promotion.html
    state = jax.lax.convert_element_type(state_inval, jnp.dtype(jnp.complex128))

    qref_set_state_p.bind(*in_qubits, state)


@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_measure_prim)
def handle_measure(self, wire, reset, postselect):
    """Handle the conversion from plxpr to Catalyst jaxpr for the mid-circuit measure primitive."""
    if isinstance(wire, DynamicJaxprTracer) and isinstance(wire.val.aval, QrefQubit):
        in_qubit = wire
    else:
        in_qubit = qref_get_p.bind(self.init_qreg, wire)

    result = qref_measure_p.bind(in_qubit, postselect=postselect)

    if reset:
        # Constants need to be passed as input values for some reason I forgot about.
        correction = jaxpr_pad_consts(
            [
                jax.make_jaxpr(lambda: qinst_p.bind(out_wire, op="PauliX", qubits_len=1))().jaxpr,
                jax.make_jaxpr(lambda: out_wire)().jaxpr,
            ]
        )
        out_wire = cond_p.bind(
            result,
            out_wire,
            out_wire,
            branch_jaxprs=correction,
            num_implicit_outputs=None,
        )[0]

    result = jnp.astype(result, int)
    return result


# pylint: disable=unused-argument, too-many-positional-arguments, too-many-arguments
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_measure_in_basis_prim)
def handle_measure_in_basis(self, angle, wire, plane, reset, postselect):
    """Handle the conversion from plxpr to Catalyst jaxpr for the measure_in_basis primitive"""
    _angle = jax.lax.convert_element_type(angle, jnp.dtype(jnp.float64))

    try:
        _plane = MeasurementPlane(plane)
    except ValueError as e:
        raise ValueError(
            f"Measurement plane must be one of {[plane.value for plane in MeasurementPlane]}"
        ) from e

    in_qreg, in_wire = (
        _[0] for _ in get_in_qubit_values([wire], self.qubit_index_recorder, self.init_qreg)
    )
    result, out_wire = measure_in_basis_p.bind(_angle, in_wire, plane=_plane, postselect=postselect)

    in_qreg[in_qreg.global_index_to_local_index(wire)] = out_wire

    return result


# pylint: disable=unused-argument
@PLxPRToQuantumJaxprInterpreter.register_primitive(plxpr_ctrl_transform_prim)
def handle_ctrl_transform(self, *invals, jaxpr, n_control, control_values, work_wires, n_consts):
    """Interpret a control transform primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:-n_control]
    control_wires = invals[-n_control:]

    unroller = copy(self)
    unroller.control_wires += tuple(control_wires)
    unroller.control_values += tuple(control_values)
    unroller.eval(jaxpr, consts, *args)
    return []


@PLxPRToQuantumJaxprInterpreter.register_primitive(transform_prim)
def _error_on_transform(*args, **kwargs):
    raise NotImplementedError("transforms cannot currently be applied inside a QNode.")


_special_op_bind_call = {
    qml.QubitUnitary: _qubit_unitary_bind_call,
    qml.GlobalPhase: _gphase_bind_call,
    qml.PCPhase: _pcphase_bind_call,
    qml.PauliRot: _pauli_rot_bind_call,
}
