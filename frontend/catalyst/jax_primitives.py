# Copyright 2022-2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains JAX-compatible quantum primitives to support the lowering
of quantum operations, measurements, and observables to JAXPR.
"""

import functools
from dataclasses import dataclass
from enum import Enum
from typing import List, Union

import jax
import pennylane as qml
from jax._src import core
from jax._src.core import pytype_aval_mappings
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.lax import _merge_dyn_shape
from jax._src.lib.mlir import ir
from jax.core import AbstractValue
from jax.extend.core import ClosedJaxpr, Primitive
from jax.interpreters import mlir

from catalyst.jax_extras.tracing import (
    DynshapePrimitive,
    cond_expansion_strategy,
    for_loop_expansion_strategy,
    infer_output_type_jaxpr,
    switch_expansion_strategy,
    while_loop_expansion_strategy,
)
from catalyst.utils.calculate_grad_shape import Signature, calculate_grad_shape

# pylint: disable=unused-argument,too-many-lines,too-many-statements,protected-access

#########
# Types #
#########


#
# qbit
#
class AbstractQbit(AbstractValue):
    """Abstract Qbit"""

    hash_value = hash("AbstractQubit")

    def __eq__(self, other):  # pragma: nocover
        return isinstance(other, AbstractQbit)

    def __hash__(self):  # pragma: nocover
        return self.hash_value


class ConcreteQbit:
    """Concrete Qbit."""


def _qbit_lowering(aval):
    assert isinstance(aval, AbstractQbit)
    return ir.OpaqueType.get("quantum", "bit")


#
# qreg
#
class AbstractQreg(AbstractValue):
    """Abstract quantum register."""

    hash_value = hash("AbstractQreg")

    def __eq__(self, other):
        return isinstance(other, AbstractQreg)

    def __hash__(self):
        return self.hash_value


class ConcreteQreg:
    """Concrete quantum register."""


def _qreg_lowering(aval):
    assert isinstance(aval, AbstractQreg)
    return ir.OpaqueType.get("quantum", "reg")


#
# observable
#
class AbstractObs(AbstractValue):
    """Abstract observable."""

    def __init__(self, num_qubits_or_qreg=None, primitive=None):
        self.num_qubits = None
        self.qreg = None
        self.primitive = primitive

        if isinstance(num_qubits_or_qreg, int):
            self.num_qubits = num_qubits_or_qreg
        elif isinstance(num_qubits_or_qreg, AbstractQreg):
            self.qreg = num_qubits_or_qreg

    def __eq__(self, other):  # pragma: nocover
        if not isinstance(other, AbstractObs):
            return False

        return (
            self.num_qubits == other.num_qubits
            and self.qreg == other.qreg
            and self.primitive == other.primitive
        )

    def __hash__(self):  # pragma: nocover
        return hash(self.primitive) + hash(self.num_qubits) + hash(self.qreg)


class ConcreteObs(AbstractObs):
    """Concrete observable."""


def _obs_lowering(aval):
    assert isinstance(aval, AbstractObs)
    return (ir.OpaqueType.get("quantum", "obs"),)


#
# registration
#
mlir.ir_type_handlers[AbstractQbit] = _qbit_lowering
mlir.ir_type_handlers[AbstractQreg] = _qreg_lowering
mlir.ir_type_handlers[AbstractObs] = _obs_lowering


class Folding(Enum):
    """
    Folding types supported by ZNE mitigation
    """

    GLOBAL = "global"
    RANDOM = "local-random"
    ALL = "local-all"


class MeasurementPlane(Enum):
    """
    Measurement planes for arbitrary-basis measurements in MBQC
    """

    XY = "XY"
    YZ = "YZ"
    ZX = "ZX"


##############
# Primitives #
##############

zne_p = Primitive("zne")
device_init_p = Primitive("device_init")
device_init_p.multiple_results = True
device_release_p = Primitive("device_release")
device_release_p.multiple_results = True
num_qubits_p = Primitive("num_qubits")
qalloc_p = Primitive("qalloc")
qdealloc_p = Primitive("qdealloc")
qdealloc_p.multiple_results = True
qdealloc_qb_p = Primitive("qdealloc_qb")
qdealloc_qb_p.multiple_results = True
qextract_p = Primitive("qextract")
qinsert_p = Primitive("qinsert")
gphase_p = Primitive("gphase")
gphase_p.multiple_results = True
qinst_p = Primitive("qinst")
qinst_p.multiple_results = True
unitary_p = Primitive("unitary")
unitary_p.multiple_results = True
pauli_rot_p = Primitive("pauli_rot")
pauli_rot_p.multiple_results = True
pauli_measure_p = Primitive("pauli_measure")
pauli_measure_p.multiple_results = True
measure_p = Primitive("measure")
measure_p.multiple_results = True
compbasis_p = Primitive("compbasis")
namedobs_p = Primitive("namedobs")
mcmobs_p = Primitive("mcmobs")
hermitian_p = Primitive("hermitian")
tensorobs_p = Primitive("tensorobs")
hamiltonian_p = Primitive("hamiltonian")
sample_p = Primitive("sample")
counts_p = Primitive("counts")
counts_p.multiple_results = True
expval_p = Primitive("expval")
var_p = Primitive("var")
probs_p = Primitive("probs")
state_p = Primitive("state")
cond_p = DynshapePrimitive("cond")
cond_p.multiple_results = True
switch_p = DynshapePrimitive("switch")
switch_p.multiple_results = True
while_p = DynshapePrimitive("while_loop")
while_p.multiple_results = True
for_p = DynshapePrimitive("for_loop")
for_p.multiple_results = True
grad_p = Primitive("grad")
grad_p.multiple_results = True
func_p = core.CallPrimitive("func")
func_p.multiple_results = True
jvp_p = Primitive("jvp")
jvp_p.multiple_results = True
vjp_p = Primitive("vjp")
vjp_p.multiple_results = True
adjoint_p = Primitive("adjoint")
adjoint_p.multiple_results = True
print_p = Primitive("debug_print")
print_p.multiple_results = True
python_callback_p = Primitive("python_callback")
python_callback_p.multiple_results = True
value_and_grad_p = Primitive("value_and_grad")
value_and_grad_p.multiple_results = True
assert_p = Primitive("assert")
assert_p.multiple_results = True
set_state_p = Primitive("state_prep")
set_state_p.multiple_results = True
set_basis_state_p = Primitive("set_basis_state")
set_basis_state_p.multiple_results = True
quantum_kernel_p = core.CallPrimitive("quantum_kernel")
quantum_kernel_p.multiple_results = True
measure_in_basis_p = Primitive("measure_in_basis")
measure_in_basis_p.multiple_results = True
decomprule_p = core.Primitive("decomposition_rule")
decomprule_p.multiple_results = True


def decomposition_rule(func=None, *, is_qreg=True, num_params=0, pauli_word=None, op_type=None):
    """
    Denotes the creation of a quantum definition in the intermediate representation.

    Args:
        func (Callable): The function defining the decomposition rule.
        is_qreg (bool): Indicates if the qubits involved in the decomposition are represented
            as a quantum register. Defaults to True.
        num_params (int): The number of parameters for the decomposition rule. Defaults to 0.
        pauli_word (str | None): The Pauli word associated with the decomposition rule.
            Defaults to None.
        op_type (str | None | Operation): The type of operation that the decomposition rule
            applies to. Defaults to None.

    .. note::

        Must be used with capture.

    **Example**

    .. code-block:: python
        from catalyst.jax_primitives import decomposition_rule
        from numpy import pi

        qp.capture.enable() # remember to enable capture


        @decomposition_rule(is_qreg=True, op_type=qp.RY) # specify the op type to decompose
        def my_decomp(angle, wires):
            qp.RX(-pi / 2, wires[0])
            qp.RZ(angle, wires[0])
            qp.RX(pi / 2, wires[0])


        @qp.qjit
        @qp.transform(pass_name="decompose-lowering") # apply decompose-lowering pass
        @qp.qnode(qp.device("lightning.qubit", wires=2))
        def my_circuit(angle: float):
            my_decomp(float, jax.core.ShapedArray((2,), int)) # apply the decomposition
            qp.RY(angle, 0)
            return qp.probs()



    >>> print(qp.specs(my_circuit)(pi))
    Device: lightning.qubit
    Device wires: 2
    Shots: Shots(total=None)
    Level: device

    Resource specifications:
      Total wire allocations: 2
      Total gates: 3
      Circuit depth: 3

      Gate types:
        RZ: 1
        RX: 2

      Measurements:
        probs(all wires): 1
    """

    assert not is_qreg or (
        is_qreg and num_params == 0
    ), "Decomposition rules with `qreg` do not require `num_params`."

    if op_type is not None and not isinstance(op_type, str):
        if issubclass(op_type, qml.operation.Operation):
            op_type = op_type.__name__
        else:
            raise ValueError("op_type must be a string or a pennylane operator.")

    if func is None:
        return functools.partial(
            decomposition_rule,
            is_qreg=is_qreg,
            num_params=num_params,
            pauli_word=pauli_word,
            op_type=op_type,
        )

    if pauli_word is not None:
        func = functools.partial(func, pauli_word=pauli_word)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # TODO change this name to op_type
        if getattr(func, "target_gate", None) is None:
            setattr(func, "target_gate", op_type)

        if pauli_word is not None:
            jaxpr = jax.make_jaxpr(func)(theta=args[0], wires=args[1], **kwargs)
        else:
            jaxpr = jax.make_jaxpr(func)(*args, **kwargs)
        decomprule_p.bind(pyfun=func, func_jaxpr=jaxpr, is_qreg=is_qreg, num_params=num_params)

    return wrapper


def _assert_jaxpr_without_constants(jaxpr: ClosedJaxpr):
    assert len(jaxpr.consts) == 0, (
        "Abstract evaluation is not defined for Jaxprs with non-empty constants because these are "
        "not available at the time of the creation of output tracers."
    )


@python_callback_p.def_abstract_eval
def _python_callback_abstract_eval(*avals, callback, custom_grad, results_aval):
    """Abstract evaluation"""
    return results_aval


#
# print
#
@print_p.def_abstract_eval
def _print_abstract_eval(*args, string=None, memref=False):
    return ()


@print_p.def_impl
def _print_def_impl(*args, string=None, memref=False):  # pragma: no cover
    print(*args)
    return ()


#
# Decomp rule
#
@decomprule_p.def_abstract_eval
def _decomposition_rule_abstract(*, pyfun, func_jaxpr, is_qreg=False, num_params=None, **params):
    return ()


#
# grad
#
@dataclass
class GradParams:
    """Common gradient parameters. The parameters are expected to be checked before the creation of
    this structure"""

    method: str
    scalar_out: bool
    h: float
    argnums: Union[int, List]
    scalar_argnums: bool = None
    expanded_argnums: List[int] = None
    with_value: bool = False  # if true it calls value_and_grad instead of grad


@grad_p.def_abstract_eval
def _grad_abstract(*args, jaxpr, fn, grad_params):
    """This function is called with abstract arguments for tracing."""
    signature = Signature(jaxpr.consts + jaxpr.in_avals, jaxpr.out_avals)
    offset = len(jaxpr.consts)
    new_argnums = [num + offset for num in grad_params.expanded_argnums]
    transformed_signature = calculate_grad_shape(signature, new_argnums)
    return tuple(transformed_signature.get_results())


# value_and_grad
#
@value_and_grad_p.def_abstract_eval
def _value_and_grad_abstract(*args, jaxpr, fn, grad_params):  # pylint: disable=unused-argument
    """This function is called with abstract arguments for tracing.
    Note: argument names must match these of `_value_and_grad_lowering`."""

    signature = Signature(jaxpr.consts + jaxpr.in_avals, jaxpr.out_avals)
    offset = len(jaxpr.consts)
    new_argnums = [num + offset for num in grad_params.expanded_argnums]
    transformed_signature = calculate_grad_shape(signature, new_argnums)
    return tuple(jaxpr.out_avals + transformed_signature.get_results())


#
# vjp/jvp
#
@jvp_p.def_abstract_eval
def _jvp_abstract(*args, jaxpr, fn, grad_params):  # pylint: disable=unused-argument
    """This function is called with abstract arguments for tracing.
    Note: argument names must match these of `_jvp_lowering`."""
    return jaxpr.out_avals + jaxpr.out_avals


@vjp_p.def_abstract_eval
# pylint: disable=unused-argument
def _vjp_abstract(*args, jaxpr, fn, grad_params):
    """This function is called with abstract arguments for tracing."""
    return jaxpr.out_avals + [jaxpr.in_avals[i] for i in grad_params.expanded_argnums]


#
# zne
#
@zne_p.def_abstract_eval
def _zne_abstract_eval(*args, folding, jaxpr, fn):  # pylint: disable=unused-argument
    shape = list(args[-1].shape)
    if len(jaxpr.out_avals) > 1:
        shape.append(len(jaxpr.out_avals))
    return core.ShapedArray(shape, jaxpr.out_avals[0].dtype)


#
# device_init
#
@device_init_p.def_abstract_eval
def _device_init_abstract_eval(shots, auto_qubit_management, rtd_lib, rtd_name, rtd_kwargs):
    return ()


#
# device_release
#
@device_release_p.def_abstract_eval
def _device_release_abstract_eval():
    return ()


#
# num_qubits_p
#
@num_qubits_p.def_abstract_eval
def _num_qubits_abstract_eval():
    return core.ShapedArray((), jax.numpy.int64)


#
# qalloc
#
@qalloc_p.def_abstract_eval
def _qalloc_abstract_eval(size):
    """This function is called with abstract arguments for tracing."""
    return AbstractQreg()


#
# qdealloc
#
@qdealloc_p.def_abstract_eval
def _qdealloc_abstract_eval(qreg):
    return ()


#
# qdealloc_qb
#
@qdealloc_qb_p.def_abstract_eval
def _qdealloc_qb_abstract_eval(qubit):
    return ()


#
# qextract
#
@qextract_p.def_abstract_eval
def _qextract_abstract_eval(qreg, qubit_idx):
    """This function is called with abstract arguments for tracing."""
    assert isinstance(qreg, AbstractQreg), f"Expected AbstractQreg(), got {qreg}"
    return AbstractQbit()


#
# qinsert
#
@qinsert_p.def_abstract_eval
def _qinsert_abstract_eval(qreg_old, qubit_idx, qubit):
    """This function is called with abstract arguments for tracing."""
    assert isinstance(qreg_old, AbstractQreg)
    assert isinstance(qubit, AbstractQbit)
    return AbstractQreg()


#
# gphase
#
@gphase_p.def_abstract_eval
def _gphase_abstract_eval(*qubits_or_params, ctrl_len=0, adjoint=False):
    # The signature here is: (using * to denote zero or more)
    # param, ctrl_qubits*, ctrl_values*
    # since gphase has no target qubits.
    param = qubits_or_params[0]
    assert not isinstance(param, AbstractQbit)
    ctrl_qubits = qubits_or_params[-2 * ctrl_len : -ctrl_len]
    for idx in range(ctrl_len):
        qubit = ctrl_qubits[idx]
        assert isinstance(qubit, AbstractQbit)
    return (AbstractQbit(),) * (ctrl_len)


#
# qinst
#
@qinst_p.def_abstract_eval
def _qinst_abstract_eval(
    *qubits_or_params, op=None, qubits_len=0, params_len=0, ctrl_len=0, adjoint=False
):
    # The signature here is: (using * to denote zero or more)
    # qubits*, params*, ctrl_qubits*, ctrl_values*
    qubits = qubits_or_params[:qubits_len]
    ctrl_qubits = qubits_or_params[-2 * ctrl_len : -ctrl_len]
    all_qubits = qubits + ctrl_qubits
    for idx in range(qubits_len + ctrl_len):
        qubit = all_qubits[idx]
        assert isinstance(qubit, AbstractQbit)
    return (AbstractQbit(),) * (qubits_len + ctrl_len)


#
# qubit unitary operation
#
@unitary_p.def_abstract_eval
def _unitary_abstract_eval(matrix, *qubits, qubits_len=0, ctrl_len=0, adjoint=False):
    for idx in range(qubits_len + ctrl_len):
        qubit = qubits[idx]
        assert isinstance(qubit, AbstractQbit)
    return (AbstractQbit(),) * (qubits_len + ctrl_len)


#
# pauli rot operation
#
# pylint: disable=unused-variable
@pauli_rot_p.def_abstract_eval
def _pauli_rot_abstract_eval(
    *qubits_and_ctrl_qubits,
    angle=None,
    pauli_word=None,
    qubits_len=0,
    params_len=0,
    ctrl_len=0,
    adjoint=False,
):
    # The signature here is: (using * to denote zero or more)
    # qubits*, params*, ctrl_qubits*, ctrl_values*
    qubits = qubits_and_ctrl_qubits[:qubits_len]
    params = qubits_and_ctrl_qubits[qubits_len : qubits_len + params_len]
    ctrl_qubits = qubits_and_ctrl_qubits[-2 * ctrl_len : -ctrl_len]
    ctrl_values = qubits_and_ctrl_qubits[-ctrl_len:]
    all_qubits = qubits + ctrl_qubits
    assert all(isinstance(qubit, AbstractQbit) for qubit in all_qubits)
    return (AbstractQbit(),) * (qubits_len + ctrl_len)


#
# pauli measure operation
#
@pauli_measure_p.def_abstract_eval
def _pauli_measure_abstract_eval(*qubits, pauli_word=None, qubits_len=0, adjoint=False):
    qubits = qubits[:qubits_len]
    assert all(isinstance(qubit, AbstractQbit) for qubit in qubits)
    # This corresponds to the measurement value and the qubits after the measurements
    return (core.ShapedArray((), bool),) + (AbstractQbit(),) * (qubits_len)


#
# measure
#
@measure_p.def_abstract_eval
def _measure_abstract_eval(qubit, postselect: int = None):
    assert isinstance(qubit, AbstractQbit)
    return core.ShapedArray((), bool), qubit


#
# arbitrary-basis measurements
#
@measure_in_basis_p.def_abstract_eval
def _measure_in_basis_abstract_eval(
    angle: float, qubit: AbstractQbit, plane: MeasurementPlane, postselect: int = None
):
    assert isinstance(qubit, AbstractQbit)
    return core.ShapedArray((), bool), qubit


#
# compbasis observable
#
@compbasis_p.def_abstract_eval
def _compbasis_abstract_eval(*qubits_or_qreg, qreg_available=False):
    if qreg_available:
        qreg = qubits_or_qreg[0]
        assert isinstance(qreg, AbstractQreg)
        return AbstractObs(qreg, compbasis_p)
    else:
        qubits = qubits_or_qreg
        for qubit in qubits:
            assert isinstance(qubit, AbstractQbit)
        return AbstractObs(len(qubits), compbasis_p)


#
# named observable
#
@namedobs_p.def_abstract_eval
def _namedobs_abstract_eval(qubit, kind):
    assert isinstance(qubit, AbstractQbit)
    return AbstractObs()


#
# mcm observable
#
@mcmobs_p.def_abstract_eval
def _mcmobs_abstract_eval(*mcms):
    return AbstractObs(len(mcms), mcmobs_p)


#
# hermitian observable
#
@hermitian_p.def_abstract_eval
def _hermitian_abstract_eval(matrix, *qubits):
    for q in qubits:
        assert isinstance(q, AbstractQbit)
    return AbstractObs()


#
# tensor observable
#
@tensorobs_p.def_abstract_eval
def _tensorobs_abstract_eval(*terms):
    for o in terms:
        assert isinstance(o, AbstractObs)
    return AbstractObs()


#
# hamiltonian observable
#
@hamiltonian_p.def_abstract_eval
def _hamiltonian_abstract_eval(coeffs, *terms):
    for o in terms:
        assert isinstance(o, AbstractObs)
    return AbstractObs()


#
# measurements
#
def custom_measurement_staging_rule(
    primitive, jaxpr_trace, obs, dtypes, *dynamic_shape, static_shape
):
    """
    In jax, the default `def_abstract_eval` method for binding primitives keeps the abstract aval in
    the dynamic shape dimension, instead of the SSA value for the shape, i.e.

    c:i64[] = ...  # to be used as the 0th dimension of value `e`
    d:AbstractObs = ...
    e:f64[ShapedArray(int64[], weak_type=True),1] = sample[num_qubits=1] d c

    which contains escaped tracers.

    To ensure that the result DShapedArray is actually constructed with the tracer value,
    we need to provide a custom staging rule for the primitive, where we manually link
    the tracer's value to the output shape. This will now correctly produce

    e:f64[c,1] = sample[num_qubits=1] d c

    This works because when jax processes a primitive during making jaxprs, the default
    is to only look at the abstract avals of the primitive. Providing a custom staging rule
    circumvents the above default logic.

    See jax._src.interpreters.partial_eval.process_primitive and default_process_primitive,
    https://github.com/jax-ml/jax/blob/a54319ec1886ed920d50cacf10e147a743888464/jax/_src/interpreters/partial_eval.py#L1881C7-L1881C24
    """

    shape = _merge_dyn_shape(static_shape, dynamic_shape)
    if not dynamic_shape:
        # Some PL transforms, like @qml.batch_params, do not support dynamic shapes yet
        # Therefore we still keep static shapes when possible
        # This can be removed, and all avals turned into DShapedArrays, when
        # dynamic program capture in PL is complete
        out_shapes = tuple(core.ShapedArray(shape, dtype) for dtype in dtypes)
    else:
        out_shapes = tuple(core.DShapedArray(shape, dtype) for dtype in dtypes)

    in_tracers = [obs] + list(dynamic_shape)

    params = {"static_shape": static_shape}

    eqn, out_tracers = jaxpr_trace.make_eqn(in_tracers, out_shapes, primitive, params, [])

    jaxpr_trace.frame.add_eqn(eqn)
    return out_tracers if len(out_tracers) > 1 else out_tracers[0]


#
# sample measurement
#
def sample_staging_rule(jaxpr_trace, _src, obs, *dynamic_shape, static_shape):
    """
    The result shape of `sample_p` is (shots, num_qubits).
    """
    shape = _merge_dyn_shape(static_shape, dynamic_shape)
    if obs.primitive is compbasis_p:
        if obs.num_qubits:
            assert isinstance(shape[1], int)
            assert shape[1] == obs.num_qubits

    return custom_measurement_staging_rule(
        sample_p,
        jaxpr_trace,
        obs,
        [jax.numpy.dtype("float64")],
        *dynamic_shape,
        static_shape=static_shape,
    )


pe.custom_staging_rules[sample_p] = sample_staging_rule


#
# counts measurement
#
def counts_staging_rule(jaxpr_trace, _src, obs, *dynamic_shape, static_shape):
    """
    The result shape of `counts_p` is (tensor<Nxf64>, tensor<Nxi64>)
    where N = 2**number_of_qubits.
    """

    shape = _merge_dyn_shape(static_shape, dynamic_shape)
    if obs.primitive in (compbasis_p, mcmobs_p):
        if obs.num_qubits:
            if isinstance(shape[0], int):
                assert shape == (2**obs.num_qubits,)
    else:
        assert shape == (2,)

    return custom_measurement_staging_rule(
        counts_p,
        jaxpr_trace,
        obs,
        [jax.numpy.dtype("float64"), jax.numpy.dtype("int64")],
        *dynamic_shape,
        static_shape=static_shape,
    )


pe.custom_staging_rules[counts_p] = counts_staging_rule


#
# expval measurement
#
@expval_p.def_abstract_eval
def _expval_abstract_eval(obs, shape=None):
    assert isinstance(obs, AbstractObs)
    return core.ShapedArray((), jax.numpy.float64)


#
# var measurement
#
@var_p.def_abstract_eval
def _var_abstract_eval(obs, shape=None):
    assert isinstance(obs, AbstractObs)
    return core.ShapedArray((), jax.numpy.float64)


#
# probs measurement
#
def probs_staging_rule(jaxpr_trace, _src, obs, *dynamic_shape, static_shape):
    """
    The result shape of probs_p is (2^num_qubits,).
    """
    return custom_measurement_staging_rule(
        probs_p,
        jaxpr_trace,
        obs,
        [jax.numpy.dtype("float64")],
        *dynamic_shape,
        static_shape=static_shape,
    )


pe.custom_staging_rules[probs_p] = probs_staging_rule


#
# state measurement
#
def state_staging_rule(jaxpr_trace, _src, obs, *dynamic_shape, static_shape):
    """
    The result shape of state_p is (2^num_qubits,).
    """
    if obs.primitive is compbasis_p:
        assert not obs.num_qubits, """
        A "wires" argument should not be provided since state() always
        returns a pure state describing all wires in the device.
        """
    else:
        raise TypeError("state only supports computational basis")

    return custom_measurement_staging_rule(
        state_p,
        jaxpr_trace,
        obs,
        [jax.numpy.dtype("complex128")],
        *dynamic_shape,
        static_shape=static_shape,
    )


pe.custom_staging_rules[state_p] = state_staging_rule


#
# cond
#
@cond_p.def_abstract_eval
def _cond_abstract_eval(*args, branch_jaxprs, num_implicit_outputs: int, **kwargs):
    out_type = infer_output_type_jaxpr(
        [()] + branch_jaxprs[0].jaxpr.invars,
        [],
        branch_jaxprs[0].jaxpr.outvars[num_implicit_outputs:],
        expansion_strategy=cond_expansion_strategy(),
        num_implicit_inputs=None,
    )
    return out_type


#
# Index Switch
#
@switch_p.def_abstract_eval
def _switch_p_abstract_eval(*args, branch_jaxprs, num_implicit_outputs: int, **kwargs):
    out_type = infer_output_type_jaxpr(
        [()] + branch_jaxprs[0].jaxpr.invars,
        [],
        branch_jaxprs[0].jaxpr.outvars[num_implicit_outputs:],
        expansion_strategy=switch_expansion_strategy(),
        num_implicit_inputs=None,
    )

    return out_type


#
# while loop
#
@while_p.def_abstract_eval
def _while_loop_abstract_eval(
    *in_type,
    body_jaxpr,
    num_implicit_inputs,
    preserve_dimensions,
    cond_nconsts,
    body_nconsts,
    **kwargs,
):
    _assert_jaxpr_without_constants(body_jaxpr)
    all_nconsts = cond_nconsts + body_nconsts
    return infer_output_type_jaxpr(
        body_jaxpr.jaxpr.invars[:all_nconsts],
        body_jaxpr.jaxpr.invars[all_nconsts:],
        body_jaxpr.jaxpr.outvars[num_implicit_inputs:],
        expansion_strategy=while_loop_expansion_strategy(preserve_dimensions),
    )


#
# for loop
#
@for_p.def_abstract_eval
def _for_loop_abstract_eval(
    *args, body_jaxpr, num_implicit_inputs, preserve_dimensions, body_nconsts, **kwargs
):
    _assert_jaxpr_without_constants(body_jaxpr)

    return infer_output_type_jaxpr(
        body_jaxpr.jaxpr.invars[:body_nconsts],
        body_jaxpr.jaxpr.invars[body_nconsts:],
        body_jaxpr.jaxpr.outvars[num_implicit_inputs:],
        expansion_strategy=for_loop_expansion_strategy(preserve_dimensions),
        num_implicit_inputs=num_implicit_inputs,
    )


#
# assert
#
@assert_p.def_abstract_eval
def _assert_abstract(assertion, error):
    return ()


#
# state_prep
#
@set_state_p.def_abstract_eval
def set_state_abstract(*qubits_or_params):
    """Abstract evaluation"""
    length = len(qubits_or_params)
    qubits_length = length - 1
    return (AbstractQbit(),) * qubits_length


#
# set_basis_state
#
@set_basis_state_p.def_abstract_eval
def set_basis_state_abstract(*qubits_or_params):
    """Abstract evaluation"""
    length = len(qubits_or_params)
    qubits_length = length - 1
    return (AbstractQbit(),) * qubits_length


#
# adjoint
#
@adjoint_p.def_abstract_eval
def _adjoint_abstract(*args, args_tree, jaxpr):
    return jaxpr.out_avals[-1:]


def _scalar_abstractify(t):
    # pylint: disable=protected-access
    if t in {int, float, complex, bool} or isinstance(t, jax._src.numpy.scalar_types._ScalarMeta):
        return core.ShapedArray([], dtype=t, weak_type=True)
    raise TypeError(f"Argument type {t} is not a valid JAX type.")


pytype_aval_mappings[type] = _scalar_abstractify
pytype_aval_mappings[jax._src.numpy.scalar_types._ScalarMeta] = _scalar_abstractify
pytype_aval_mappings[ConcreteQbit] = lambda _: AbstractQbit()
pytype_aval_mappings[ConcreteQreg] = lambda _: AbstractQreg()
