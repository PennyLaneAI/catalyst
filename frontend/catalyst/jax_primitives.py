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

import copy
import functools
import sys
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import Iterable, List, Union

import jax
import numpy as np
import pennylane as qml
from jax._src import core, source_info_util, util
from jax._src.core import pytype_aval_mappings
from jax._src.interpreters import partial_eval as pe
from jax._src.lax.lax import _merge_dyn_shape, _nary_lower_hlo, cos_p, sin_p
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.pjit import _pjit_lowering, jit_p
from jax.core import AbstractValue
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jax.tree_util import PyTreeDef, tree_unflatten
from jaxlib.mlir._mlir_libs import _mlir as _ods_cext
from jaxlib.mlir.dialects.arith import (
    AddIOp,
    CeilDivSIOp,
    ConstantOp,
    ExtUIOp,
    IndexCastOp,
    MulIOp,
    SubIOp,
)
from jaxlib.mlir.dialects.func import FunctionType
from jaxlib.mlir.dialects.scf import ConditionOp, ForOp, IfOp, IndexSwitchOp, WhileOp, YieldOp
from jaxlib.mlir.dialects.stablehlo import ConstantOp as StableHLOConstantOp
from jaxlib.mlir.dialects.stablehlo import ConvertOp as StableHLOConvertOp

# TODO: remove after jax v0.7.2 upgrade
# Mock _ods_cext.globals.register_traceback_file_exclusion due to API conflicts between
# Catalyst's MLIR version and the MLIR version used by JAX. The current JAX version has not
# yet updated to the latest MLIR, causing compatibility issues. This workaround will be removed
# once JAX updates to a compatible MLIR version
# pylint: disable=ungrouped-imports
from catalyst.jax_extras.patches import mock_attributes
from catalyst.utils.patching import Patcher

with Patcher(
    (
        _ods_cext,
        "globals",
        mock_attributes(
            # pylint: disable=c-extension-no-member
            _ods_cext.globals,
            {"register_traceback_file_exclusion": lambda x: None},
        ),
    ),
):
    from mlir_quantum.dialects.catalyst import (
        AssertionOp,
        CallbackCallOp,
        CallbackOp,
        PrintOp,
    )
    from mlir_quantum.dialects.gradient import (
        CustomGradOp,
        ForwardOp,
        GradOp,
        JVPOp,
        ReverseOp,
        ValueAndGradOp,
        VJPOp,
    )
    from mlir_quantum.dialects.mbqc import MeasureInBasisOp
    from mlir_quantum.dialects.mitigation import ZneOp
    from mlir_quantum.dialects.qec import PPMeasurementOp
    from mlir_quantum.dialects.quantum import (
        AdjointOp,
        AllocOp,
        ComputationalBasisOp,
        CountsOp,
        CustomOp,
        DeallocOp,
        DeallocQubitOp,
        DeviceInitOp,
        DeviceReleaseOp,
        ExpvalOp,
        ExtractOp,
        GlobalPhaseOp,
        HamiltonianOp,
        HermitianOp,
        InsertOp,
        MeasureOp,
        MultiRZOp,
        NamedObsOp,
        NumQubitsOp,
        PauliRotOp,
        PCPhaseOp,
        ProbsOp,
        QubitUnitaryOp,
        SampleOp,
        SetBasisStateOp,
        SetStateOp,
        StateOp,
        TensorOp,
        VarianceOp,
    )
    from mlir_quantum.dialects.quantum import YieldOp as QYieldOp
    from catalyst.jax_primitives_utils import (
        ApplyRegisteredPassOp,
        cache,
        create_call_op,
        get_cached,
        get_call_jaxpr,
        get_symbolref,
        lower_callable,
        lower_jaxpr,
    )

from pennylane.capture.primitives import jacobian_prim as pl_jac_prim

from catalyst.compiler import get_lib_path
from catalyst.jax_extras import (
    ClosedJaxpr,
    DynshapePrimitive,
    cond_expansion_strategy,
    for_loop_expansion_strategy,
    infer_output_type_jaxpr,
    switch_expansion_strategy,
    while_loop_expansion_strategy,
)
from catalyst.utils.calculate_grad_shape import Signature, calculate_grad_shape
from catalyst.utils.exceptions import CompileError
from catalyst.utils.extra_bindings import FromElementsOp, TensorExtractOp
from catalyst.utils.types import convert_shaped_arrays_to_tensors

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

quantum_subroutine_p = copy.deepcopy(jit_p)
quantum_subroutine_p.name = "quantum_subroutine_p"
subroutine_cache: dict[callable, callable] = {}


def subroutine(func):
    """
    Denotes the creation of a function in the intermediate representation.

    May be used to reduce compilation times. Instead of repeatedly compiling
    inlined versions of the function passed as a parameter, when functions
    are annotated with a subroutine, a single version of the function
    will be compiled and called from potentially multiple callsites.

    .. note::

        Subroutines are only available when using the PLxPR program capture
        interface.


    **Example**

    .. code-block:: python

        @subroutine
        def Hadamard_on_wire_0():
            qml.Hadamard(0)

        qml.capture.enable()

        @qjit
        @qml.qnode(dev)
        def main():
            Hadamard_on_wire_0()
            Hadamard_on_wire_0()
            return qml.state()

        print(main.mlir)
        qml.capture.disable()
    """

    # pylint: disable-next=import-outside-toplevel
    from catalyst.api_extensions.callbacks import WRAPPER_ASSIGNMENTS

    old_jit_p = jax._src.pjit.jit_p

    @functools.wraps(func, assigned=WRAPPER_ASSIGNMENTS)
    def inside(*args, **kwargs):
        with Patcher(
            (
                jax._src.pjit,
                "jit_p",
                old_jit_p,
            ),
        ):
            return func(*args, **kwargs)

    @functools.wraps(inside, assigned=WRAPPER_ASSIGNMENTS)
    def wrapper(*args, **kwargs):

        if not qml.capture.enabled():
            msg = "Subroutine is only available with capture enabled."
            raise CompileError(msg)

        with Patcher(
            (
                jax._src.pjit,
                "jit_p",
                quantum_subroutine_p,
            ),
        ):
            return jax.jit(inside)(*args, **kwargs)

    return wrapper


def decomposition_rule(func=None, *, is_qreg=True, num_params=0, pauli_word=None):
    """
    Denotes the creation of a quantum definition in the intermediate representation.
    """

    assert not is_qreg or (
        is_qreg and num_params == 0
    ), "Decomposition rules with `qreg` do not require `num_params`."

    if func is None:
        return functools.partial(decomposition_rule, is_qreg=is_qreg, num_params=num_params)

    if pauli_word is not None:
        func = functools.partial(func, pauli_word=pauli_word)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
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


@python_callback_p.def_impl
def _python_callback_def_impl(*avals, callback, custom_grad, results_aval):  # pragma: no cover
    """Concrete evaluation"""
    raise NotImplementedError()


def _python_callback_lowering(
    jax_ctx: mlir.LoweringRuleContext, *args, callback, custom_grad, results_aval
):
    """Callback lowering"""

    sys.path.append(get_lib_path("runtime", "RUNTIME_LIB_DIR"))
    import catalyst_callback_registry as registry  # pylint: disable=import-outside-toplevel

    callback_id = registry.register(callback)

    params_ty = [arg.type for arg in args]
    results_ty = list(convert_shaped_arrays_to_tensors(results_aval))
    fn_ty = FunctionType.get(inputs=params_ty, results=results_ty)
    fn_ty_attr = ir.TypeAttr.get(fn_ty)
    cache_key = (callback_id, *params_ty, *results_ty)
    if callbackOp := get_cached(jax_ctx, cache_key):
        symbol = callbackOp.sym_name.value
        symbol_attr = ir.FlatSymbolRefAttr.get(symbol)
        return CallbackCallOp(results_ty, symbol_attr, args).results

    module = jax_ctx.module_context.module
    ip = module.body
    attrs = [fn_ty_attr, callback_id, len(args), len(results_ty)]
    with ir.InsertionPoint(ip):
        # TODO: Name mangling for callbacks
        name = callback.__name__
        callbackOp = CallbackOp(f"callback_{name}_{callback_id}", *attrs)

    cache(jax_ctx, cache_key, callbackOp)
    symbol = callbackOp.sym_name.value
    symbol_attr = ir.FlatSymbolRefAttr.get(symbol)
    retval = CallbackCallOp(results_ty, symbol_attr, args).results

    if not custom_grad:
        return retval

    assert custom_grad._fwd and custom_grad._bwd
    fwd = custom_grad._fwd
    rev = custom_grad._bwd
    fwd_jaxpr = custom_grad._fwd_jaxpr
    rev_jaxpr = custom_grad._bwd_jaxpr
    mlir_fwd = lower_callable(jax_ctx, fwd, fwd_jaxpr)
    mlir_rev = lower_callable(jax_ctx, rev, rev_jaxpr)
    sym_fwd = mlir_fwd.sym_name.value + ".fwd"

    argc = len(args)
    resc = len(results_ty)
    len_tape = len(mlir_fwd.type.results) - resc

    # args_ty = inputs and cotangents since they are shadows
    args_ty = [arg.type for arg in args]
    # results_ty = output and cotangent
    output_ty = results_ty
    # the tape is found in the mlir_fwd.type
    tape_ty = mlir_fwd.type.results[-len_tape:] if len_tape > 0 else []

    fn_fwd_ty = FunctionType.get(inputs=args_ty, results=output_ty + tape_ty)
    fn_rev_ty = FunctionType.get(inputs=output_ty + tape_ty, results=args_ty)

    fwd_fn_ty_attr = ir.TypeAttr.get(fn_fwd_ty)
    fwd_callee_attr = ir.FlatSymbolRefAttr.get(mlir_fwd.sym_name.value)
    sym_rev = mlir_rev.sym_name.value + ".rev"
    rev_fn_ty_attr = ir.TypeAttr.get(fn_rev_ty)
    rev_callee_attr = ir.FlatSymbolRefAttr.get(mlir_rev.sym_name.value)

    with ir.InsertionPoint(ip):
        forward = ForwardOp(sym_fwd, fwd_fn_ty_attr, fwd_callee_attr, argc, resc, len_tape)
        reverse = ReverseOp(sym_rev, rev_fn_ty_attr, rev_callee_attr, argc, resc, len_tape)
        fwd_sym_attr = ir.FlatSymbolRefAttr.get(forward.sym_name.value)
        rev_sym_attr = ir.FlatSymbolRefAttr.get(reverse.sym_name.value)
        CustomGradOp(symbol_attr, fwd_sym_attr, rev_sym_attr)

    return retval


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


def _print_lowering(jax_ctx: mlir.LoweringRuleContext, *args, string=None, memref=False):
    val = args[0] if args else None
    return PrintOp(val=val, const_val=None, print_descriptor=memref).results


#
# module
#
@quantum_kernel_p.def_impl
def _quantum_kernel_def_impl(*args, call_jaxpr, qnode, pipeline=None):  # pragma: no cover
    raise NotImplementedError()


def _quantum_kernel_lowering(ctx, *args, call_jaxpr, qnode, pipeline=None):
    """Lower's qnodes to moduleOp

    Args:
      ctx: LoweringRuleContext
      *args: List[mlir.Value] corresponding to argument values
      qnode: qml.Qnode
      call_jaxpr: jaxpr representing fn
    Returns:
      List[mlir.Value] corresponding
    """
    assert isinstance(qnode, qml.QNode), "This function expects qnodes"
    if pipeline is None:
        pipeline = tuple()

    func_op = lower_callable(ctx, qnode, call_jaxpr, pipeline)
    call_op = create_call_op(ctx, func_op, *args)
    return call_op.results


@func_p.def_impl
def _func_def_impl(*args, call_jaxpr, fn):  # pragma: no cover
    raise NotImplementedError()


def _func_lowering(ctx, *args, call_jaxpr, fn):
    """Lower a quantum function into MLIR in a two step process.
    The first step is the compilation of the definition of the function fn.
    The second step is compiling a call to function fn.

    Args:
      ctx: the MLIR context
      args: list of arguments or abstract arguments to the function
      name: name of the function
      call_jaxpr: the jaxpr representation of the fn
      fn: the function being compiled
    """
    func_op = lower_callable(ctx, fn, call_jaxpr)
    call_op = create_call_op(ctx, func_op, *args)
    return call_op.results


#
# Decomp rule
#
@decomprule_p.def_abstract_eval
def _decomposition_rule_abstract(*, pyfun, func_jaxpr, is_qreg=False, num_params=None, **params):
    return ()


def _decomposition_rule_lowering(ctx, *, pyfun, func_jaxpr, **params):
    """Lower a quantum decomposition rule into MLIR in a single step process.
    The step is the compilation of the definition of the function fn.
    """

    # Set the visibility of the decomposition rule to public
    # to avoid the elimination by the compiler
    lower_callable(ctx, pyfun, func_jaxpr, public=True, **params)
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


@grad_p.def_impl
def _grad_def_impl(ctx, *args, jaxpr, fn, grad_params):  # pragma: no cover
    raise NotImplementedError()


@grad_p.def_abstract_eval
def _grad_abstract(*args, jaxpr, fn, grad_params):
    """This function is called with abstract arguments for tracing."""
    signature = Signature(jaxpr.consts + jaxpr.in_avals, jaxpr.out_avals)
    offset = len(jaxpr.consts)
    new_argnums = [num + offset for num in grad_params.expanded_argnums]
    transformed_signature = calculate_grad_shape(signature, new_argnums)
    return tuple(transformed_signature.get_results())


def _grad_lowering(ctx, *args, jaxpr, fn, grad_params):
    """Lowering function to gradient.
    Args:
        ctx: the MLIR context
        args: the points in the function in which we are to calculate the derivative
        jaxpr: the jaxpr representation of the grad op
        fn (GradCallable): the function to be differentiated
        method: the method used for differentiation
        h: the difference for finite difference. May be None when fn is not finite difference.
        argnums: argument indices which define over which arguments to
            differentiate.
    """
    consts = []
    offset = len(args) - len(jaxpr.consts)
    for i, jax_array_or_tracer in enumerate(jaxpr.consts):
        if isinstance(jax_array_or_tracer, jax._src.interpreters.partial_eval.DynamicJaxprTracer):
            # There are some cases where this value cannot be converted into
            # a jax.numpy.array.
            # in that case we get it from the arguments.
            consts.append(args[offset + i])
        else:
            # ``ir.DenseElementsAttr.get()`` constructs a dense elements attribute from an array of
            # element values. This doesn't support ``jaxlib.xla_extension.Array``, so we have to
            # cast such constants to numpy array types.
            const = jax_array_or_tracer
            const_type = ir.RankedTensorType.get(const.shape, mlir.dtype_to_ir_type(const.dtype))
            nparray = np.asarray(const)
            attr = ir.DenseElementsAttr.get(nparray, type=const_type)
            constval = StableHLOConstantOp(attr).results
            consts.append(constval)

    method, h, argnums = grad_params.method, grad_params.h, grad_params.expanded_argnums
    mlir_ctx = ctx.module_context.context
    finiteDiffParam = None
    if h:
        f64 = ir.F64Type.get(mlir_ctx)
        finiteDiffParam = ir.FloatAttr.get(f64, h)
    offset = len(jaxpr.consts)
    new_argnums = [num + offset for num in argnums]
    argnum_numpy = np.array(new_argnums)
    diffArgIndices = ir.DenseIntElementsAttr.get(argnum_numpy)
    func_op = lower_jaxpr(ctx, jaxpr, (method, h, *argnums))
    symbol_ref = get_symbolref(ctx, func_op)
    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)

    len_args = len(args)
    index = len_args - len(consts)
    args_and_consts = consts + list(args[:index])

    return GradOp(
        flat_output_types,
        ir.StringAttr.get(method),
        symbol_ref,
        mlir.flatten_ir_values(args_and_consts),
        diffArgIndices=diffArgIndices,
        finiteDiffParam=finiteDiffParam,
    ).results


# pylint: disable=too-many-arguments
def _capture_grad_lowering(ctx, *args, argnums, jaxpr, n_consts, method, h, fn, scalar_out):
    mlir_ctx = ctx.module_context.context
    f64 = ir.F64Type.get(mlir_ctx)
    finiteDiffParam = ir.FloatAttr.get(f64, h)

    new_argnums = [num + n_consts for num in argnums]
    argnum_numpy = np.array(new_argnums)
    diffArgIndices = ir.DenseIntElementsAttr.get(argnum_numpy)
    func_op = lower_jaxpr(ctx, jaxpr, (method, h, *new_argnums), fn=fn)
    symbol_ref = get_symbolref(ctx, func_op)
    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)

    return GradOp(
        flat_output_types,
        ir.StringAttr.get(method),
        symbol_ref,
        mlir.flatten_ir_values(args),
        diffArgIndices=diffArgIndices,
        finiteDiffParam=finiteDiffParam,
    ).results


# value_and_grad
#
@value_and_grad_p.def_impl
def _value_and_grad_def_impl(ctx, *args, jaxpr, fn, grad_params):  # pragma: no cover
    raise NotImplementedError()


@value_and_grad_p.def_abstract_eval
def _value_and_grad_abstract(*args, jaxpr, fn, grad_params):  # pylint: disable=unused-argument
    """This function is called with abstract arguments for tracing.
    Note: argument names must match these of `_value_and_grad_lowering`."""

    signature = Signature(jaxpr.consts + jaxpr.in_avals, jaxpr.out_avals)
    offset = len(jaxpr.consts)
    new_argnums = [num + offset for num in grad_params.expanded_argnums]
    transformed_signature = calculate_grad_shape(signature, new_argnums)
    return tuple(jaxpr.out_avals + transformed_signature.get_results())


def _value_and_grad_lowering(ctx, *args, jaxpr, fn, grad_params):
    """
    Returns:
        MLIR results
    """
    consts = []
    offset = len(args) - len(jaxpr.consts)
    for i, jax_array_or_tracer in enumerate(jaxpr.consts):
        if not isinstance(
            jax_array_or_tracer, jax._src.interpreters.partial_eval.DynamicJaxprTracer
        ):
            # ``ir.DenseElementsAttr.get()`` constructs a dense elements attribute from an array of
            # element values. This doesn't support ``jaxlib.xla_extension.Array``, so we have to
            # cast such constants to numpy array types.
            const = jax_array_or_tracer
            const_type = ir.RankedTensorType.get(const.shape, mlir.dtype_to_ir_type(const.dtype))
            nparray = np.asarray(const)
            attr = ir.DenseElementsAttr.get(nparray, type=const_type)
            constval = StableHLOConstantOp(attr).results
            consts.append(constval)
        else:
            # There are some cases where this value cannot be converted into
            # a jax.numpy.array.
            # in that case we get it from the arguments.
            consts.append(args[offset + i])

    len_args = len(args)
    index = len_args - len(consts)
    args = list(args[0:index])
    method, h, argnums = grad_params.method, grad_params.h, grad_params.expanded_argnums
    mlir_ctx = ctx.module_context.context
    new_argnums = np.array([len(jaxpr.consts) + num for num in argnums])

    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)

    constants = consts

    consts_and_args = constants + args
    func_call_jaxpr = get_call_jaxpr(jaxpr)
    func_args = consts_and_args[: len(func_call_jaxpr.invars)]
    val_result_types = flat_output_types[: len(flat_output_types) - len(argnums)]
    gradient_result_types = flat_output_types[len(flat_output_types) - len(argnums) :]

    func_op = lower_jaxpr(ctx, jaxpr, (method, h, *argnums))

    symbol_ref = get_symbolref(ctx, func_op)
    return ValueAndGradOp(
        val_result_types,
        gradient_result_types,
        ir.StringAttr.get(method),
        symbol_ref,
        mlir.flatten_ir_values(func_args),
        diffArgIndices=ir.DenseIntElementsAttr.get(new_argnums),
        finiteDiffParam=ir.FloatAttr.get(ir.F64Type.get(mlir_ctx), h) if h else None,
    ).results


#
# vjp/jvp
#
@jvp_p.def_impl
def _jvp_def_impl(ctx, *args, jaxpr, fn, grad_params):  # pragma: no cover
    raise NotImplementedError()


@jvp_p.def_abstract_eval
def _jvp_abstract(*args, jaxpr, fn, grad_params):  # pylint: disable=unused-argument
    """This function is called with abstract arguments for tracing.
    Note: argument names must match these of `_jvp_lowering`."""
    return jaxpr.out_avals + jaxpr.out_avals


def _jvp_lowering(ctx, *args, jaxpr, fn, grad_params):
    """
    Returns:
        MLIR results
    """
    args = list(args)
    method, h, argnums = grad_params.method, grad_params.h, grad_params.expanded_argnums
    mlir_ctx = ctx.module_context.context
    new_argnums = np.array([len(jaxpr.consts) + num for num in argnums])

    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)
    constants = [
        StableHLOConstantOp(ir.DenseElementsAttr.get(np.asarray(const))).results
        for const in jaxpr.consts
    ]
    consts_and_args = constants + args
    func_call_jaxpr = get_call_jaxpr(jaxpr)
    func_args = consts_and_args[: len(func_call_jaxpr.invars)]
    tang_args = consts_and_args[len(func_call_jaxpr.invars) :]

    func_op = lower_jaxpr(ctx, jaxpr, (method, h, *argnums))

    assert (
        len(flat_output_types) % 2 == 0
    ), f"The total number of result tensors is expected to be even, not {len(flat_output_types)}"
    symbol_ref = get_symbolref(ctx, func_op)
    return JVPOp(
        flat_output_types[: len(flat_output_types) // 2],
        flat_output_types[len(flat_output_types) // 2 :],
        ir.StringAttr.get(method),
        symbol_ref,
        mlir.flatten_ir_values(func_args),
        mlir.flatten_ir_values(tang_args),
        diffArgIndices=ir.DenseIntElementsAttr.get(new_argnums),
        finiteDiffParam=ir.FloatAttr.get(ir.F64Type.get(mlir_ctx), h) if h else None,
    ).results


@vjp_p.def_impl
def _vjp_def_impl(ctx, *args, jaxpr, fn, grad_params):  # pragma: no cover
    raise NotImplementedError()


@vjp_p.def_abstract_eval
# pylint: disable=unused-argument
def _vjp_abstract(*args, jaxpr, fn, grad_params):
    """This function is called with abstract arguments for tracing."""
    return jaxpr.out_avals + [jaxpr.in_avals[i] for i in grad_params.expanded_argnums]


def _vjp_lowering(ctx, *args, jaxpr, fn, grad_params):
    """
    Returns:
        MLIR results
    """
    args = list(args)
    method, h, argnums = grad_params.method, grad_params.h, grad_params.expanded_argnums
    mlir_ctx = ctx.module_context.context
    new_argnums = np.array([len(jaxpr.consts) + num for num in argnums])

    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)
    constants = [
        StableHLOConstantOp(ir.DenseElementsAttr.get(np.asarray(const))).results
        for const in jaxpr.consts
    ]
    consts_and_args = constants + args
    func_call_jaxpr = get_call_jaxpr(jaxpr)
    func_args = consts_and_args[: len(func_call_jaxpr.invars)]
    cotang_args = consts_and_args[len(func_call_jaxpr.invars) :]
    func_result_types = flat_output_types[: len(flat_output_types) - len(argnums)]
    vjp_result_types = flat_output_types[len(flat_output_types) - len(argnums) :]

    func_op = lower_jaxpr(ctx, jaxpr, (method, h, *argnums))

    symbol_ref = get_symbolref(ctx, func_op)
    return VJPOp(
        func_result_types,
        vjp_result_types,
        ir.StringAttr.get(method),
        symbol_ref,
        mlir.flatten_ir_values(func_args),
        mlir.flatten_ir_values(cotang_args),
        diffArgIndices=ir.DenseIntElementsAttr.get(new_argnums),
        finiteDiffParam=ir.FloatAttr.get(ir.F64Type.get(mlir_ctx), h) if h else None,
    ).results


#
# zne
#


@zne_p.def_impl
def _zne_def_impl(ctx, *args, folding, jaxpr, fn):  # pragma: no cover
    raise NotImplementedError()


@zne_p.def_abstract_eval
def _zne_abstract_eval(*args, folding, jaxpr, fn):  # pylint: disable=unused-argument
    shape = list(args[-1].shape)
    if len(jaxpr.out_avals) > 1:
        shape.append(len(jaxpr.out_avals))
    return core.ShapedArray(shape, jaxpr.out_avals[0].dtype)


def _folding_attribute(ctx, folding):
    ctx = ctx.module_context.context
    return ir.OpaqueAttr.get(
        "mitigation",
        ("folding " + Folding(folding).name.lower()).encode("utf-8"),
        ir.NoneType.get(ctx),
        ctx,
    )


def _zne_lowering(ctx, *args, folding, jaxpr, fn):
    """Lowering function to the ZNE opearation.
    Args:
        ctx: the MLIR context
        args: the arguments with scale factors as last
        jaxpr: the jaxpr representation of the circuit
        fn: the function to be mitigated
    """
    func_op = lower_jaxpr(ctx, jaxpr)
    symbol_ref = get_symbolref(ctx, func_op)
    output_types = list(map(mlir.aval_to_ir_types, ctx.avals_out))
    flat_output_types = util.flatten(output_types)
    num_folds = args[-1]

    constants = []
    for const in jaxpr.consts:
        const_type = ir.RankedTensorType.get(const.shape, mlir.dtype_to_ir_type(const.dtype))
        nparray = np.asarray(const)
        if const.dtype == bool:
            nparray = np.packbits(nparray, bitorder="little")
        attr = ir.DenseElementsAttr.get(nparray, type=const_type)
        constantVals = StableHLOConstantOp(attr).results
        constants.append(constantVals)

    args_and_consts = constants + list(args[0:-1])

    return ZneOp(
        flat_output_types,
        symbol_ref,
        mlir.flatten_ir_values(args_and_consts),
        _folding_attribute(ctx, folding),
        num_folds,
    ).results


#
# device_init
#
@device_init_p.def_abstract_eval
def _device_init_abstract_eval(shots, auto_qubit_management, rtd_lib, rtd_name, rtd_kwargs):
    return ()


# pylint: disable=too-many-arguments, too-many-positional-arguments
def _device_init_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    shots: ir.Value,
    auto_qubit_management,
    rtd_lib,
    rtd_name,
    rtd_kwargs,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    shots_value = TensorExtractOp(ir.IntegerType.get_signless(64, ctx), shots, []).result
    DeviceInitOp(
        ir.StringAttr.get(rtd_lib),
        ir.StringAttr.get(rtd_name),
        ir.StringAttr.get(rtd_kwargs),
        shots=shots_value,
        auto_qubit_management=auto_qubit_management,
    )

    return ()


#
# device_release
#
@device_release_p.def_abstract_eval
def _device_release_abstract_eval():
    return ()


def _device_release_lowering(jax_ctx: mlir.LoweringRuleContext):
    DeviceReleaseOp()  # end of qnode
    return ()


#
# num_qubits_p
#
@num_qubits_p.def_abstract_eval
def _num_qubits_abstract_eval():
    return core.ShapedArray((), jax.numpy.int64)


def _num_qubits_lowering(jax_ctx: mlir.LoweringRuleContext):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.IntegerType.get_signless(64, ctx)
    nqubits = NumQubitsOp(result_type).result

    result_from_elements_op = ir.RankedTensorType.get((), result_type)
    from_elements_op = FromElementsOp(result_from_elements_op, nqubits)
    return from_elements_op.results


#
# qalloc
#
@qalloc_p.def_impl
def _qalloc_def_impl(ctx, size_value):  # pragma: no cover
    raise NotImplementedError()


@qalloc_p.def_abstract_eval
def _qalloc_abstract_eval(size):
    """This function is called with abstract arguments for tracing."""
    return AbstractQreg()


def _qalloc_lowering(jax_ctx: mlir.LoweringRuleContext, size_value: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    qreg_type = ir.OpaqueType.get("quantum", "reg", ctx)
    if isinstance(size_value.owner, ir.Operation) and size_value.owner.name == "stablehlo.constant":
        size_value_attr = size_value.owner.attributes["value"]
        assert ir.DenseIntElementsAttr.isinstance(size_value_attr)
        size = ir.DenseIntElementsAttr(size_value_attr)[0]
        assert size >= 0

        size_attr = ir.IntegerAttr.get(ir.IntegerType.get_signless(64, ctx), size)
        return AllocOp(qreg_type, nqubits_attr=size_attr).results
    else:
        size_value = extract_scalar(size_value, "qalloc")
        return AllocOp(qreg_type, nqubits=size_value).results


#
# qdealloc
#
@qdealloc_p.def_impl
def _qdealloc_def_impl(ctx, size_value):  # pragma: no cover
    raise NotImplementedError()


@qdealloc_p.def_abstract_eval
def _qdealloc_abstract_eval(qreg):
    return ()


def _qdealloc_lowering(jax_ctx: mlir.LoweringRuleContext, qreg):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    DeallocOp(qreg)
    return ()


#
# qdealloc_qb
#
@qdealloc_qb_p.def_abstract_eval
def _qdealloc_qb_abstract_eval(qubit):
    return ()


def _qdealloc_qb_lowering(jax_ctx: mlir.LoweringRuleContext, qubit):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    DeallocQubitOp(qubit)
    return ()


#
# qextract
#
@qextract_p.def_impl
def _qextract_def_impl(ctx, qreg, qubit_idx):  # pragma: no cover
    raise NotImplementedError()


@qextract_p.def_abstract_eval
def _qextract_abstract_eval(qreg, qubit_idx):
    """This function is called with abstract arguments for tracing."""
    assert isinstance(qreg, AbstractQreg), f"Expected AbstractQreg(), got {qreg}"
    return AbstractQbit()


def _qextract_lowering(jax_ctx: mlir.LoweringRuleContext, qreg: ir.Value, qubit_idx: ir.Value):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qreg.type), qreg.type
    assert ir.OpaqueType(qreg.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qreg.type).data == "reg"

    qubit_idx = extract_scalar(qubit_idx, "wires", "index")
    if not ir.IntegerType.isinstance(qubit_idx.type):
        raise TypeError(f"Operator wires expected to be integers, got {qubit_idx.type}!")

    if ir.IntegerType(qubit_idx.type).width < 64:
        qubit_idx = ExtUIOp(ir.IntegerType.get_signless(64), qubit_idx).result
    elif not ir.IntegerType(qubit_idx.type).width == 64:
        raise TypeError(f"Operator wires expected to be 64-bit integers, got {qubit_idx.type}!")

    qubit_type = ir.OpaqueType.get("quantum", "bit", ctx)
    return ExtractOp(qubit_type, qreg, idx=qubit_idx).results


#
# qinsert
#
@qinsert_p.def_impl
def _qinsert_def_impl(ctx, qreg_old, qubit_idx, qubit):  # pragma: no cover
    raise NotImplementedError()


@qinsert_p.def_abstract_eval
def _qinsert_abstract_eval(qreg_old, qubit_idx, qubit):
    """This function is called with abstract arguments for tracing."""
    assert isinstance(qreg_old, AbstractQreg)
    assert isinstance(qubit, AbstractQbit)
    return AbstractQreg()


def _qinsert_lowering(
    jax_ctx: mlir.LoweringRuleContext, qreg_old: ir.Value, qubit_idx: ir.Value, qubit: ir.Value
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qreg_old.type)
    assert ir.OpaqueType(qreg_old.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qreg_old.type).data == "reg"

    qubit_idx = extract_scalar(qubit_idx, "wires", "index")
    if not ir.IntegerType.isinstance(qubit_idx.type):
        raise TypeError(f"Operator wires expected to be integers, got {qubit_idx.type}!")

    if ir.IntegerType(qubit_idx.type).width < 64:
        qubit_idx = ExtUIOp(ir.IntegerType.get_signless(64), qubit_idx).result
    elif not ir.IntegerType(qubit_idx.type).width == 64:
        raise TypeError(f"Operator wires expected to be 64-bit integers, got {qubit_idx.type}!")

    qreg_type = ir.OpaqueType.get("quantum", "reg", ctx)
    return InsertOp(qreg_type, qreg_old, qubit, idx=qubit_idx).results


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


@gphase_p.def_impl
def _gphase_def_impl(*args, **kwargs):
    """Not implemented"""
    raise NotImplementedError()


def _gphase_lowering(
    jax_ctx: mlir.LoweringRuleContext, *qubits_or_params, ctrl_len=0, adjoint=False
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    param = qubits_or_params[0]
    ctrl_qubits = qubits_or_params[1 : 1 + ctrl_len]
    ctrl_values = qubits_or_params[1 + ctrl_len :]

    param = safe_cast_to_f64(param, "GlobalPhase")
    param = extract_scalar(param, "GlobalPhase")

    assert ir.F64Type.isinstance(
        param.type
    ), "Only scalar double parameters are allowed for quantum gates!"

    ctrl_values_i1 = []
    for v in ctrl_values:
        p = TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result
        ctrl_values_i1.append(p)

    GlobalPhaseOp(
        params=param,
        out_ctrl_qubits=[qubit.type for qubit in ctrl_qubits],
        in_ctrl_qubits=ctrl_qubits,
        in_ctrl_values=ctrl_values_i1,
        adjoint=adjoint,
    )
    return ctrl_qubits


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


@qinst_p.def_impl
def _qinst_def_impl(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError()


# pylint: disable=too-many-arguments
def _qinst_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *qubits_or_params,
    op=None,
    qubits_len=0,
    params_len=0,
    ctrl_len=0,
    adjoint=False,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    qubits = qubits_or_params[:qubits_len]
    params = qubits_or_params[qubits_len : qubits_len + params_len]
    ctrl_qubits = qubits_or_params[qubits_len + params_len : qubits_len + params_len + ctrl_len]
    ctrl_values = qubits_or_params[qubits_len + params_len + ctrl_len :]

    for qubit in qubits:
        assert ir.OpaqueType.isinstance(qubit.type)
        assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
        assert ir.OpaqueType(qubit.type).data == "bit"

    float_params = []
    for p in params:
        p = safe_cast_to_f64(p, op)
        p = extract_scalar(p, op)

        assert ir.F64Type.isinstance(
            p.type
        ), "Only scalar double parameters are allowed for quantum gates!"

        float_params.append(p)

    ctrl_values_i1 = []
    for v in ctrl_values:
        p = TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result
        ctrl_values_i1.append(p)

    name_attr = ir.StringAttr.get(op)
    name_str = str(name_attr)
    name_str = name_str.replace('"', "")

    if name_str == "MultiRZ":
        assert len(float_params) == 1, "MultiRZ takes one float parameter"
        float_param = float_params[0]
        return MultiRZOp(
            out_qubits=[qubit.type for qubit in qubits],
            out_ctrl_qubits=[qubit.type for qubit in ctrl_qubits],
            theta=float_param,
            in_qubits=qubits,
            in_ctrl_qubits=ctrl_qubits,
            in_ctrl_values=ctrl_values_i1,
            adjoint=adjoint,
        ).results

    if name_str == "PCPhase":
        assert len(float_params) == 2, "PCPhase takes two float parameters"
        float_param = float_params[0]
        dim_param = float_params[1]
        return PCPhaseOp(
            out_qubits=[qubit.type for qubit in qubits],
            out_ctrl_qubits=[qubit.type for qubit in ctrl_qubits],
            theta=float_param,
            dim=dim_param,
            in_qubits=qubits,
            in_ctrl_qubits=ctrl_qubits,
            in_ctrl_values=ctrl_values_i1,
            adjoint=adjoint,
        ).results

    return CustomOp(
        out_qubits=[qubit.type for qubit in qubits],
        out_ctrl_qubits=[qubit.type for qubit in ctrl_qubits],
        params=float_params,
        in_qubits=qubits,
        gate_name=name_attr,
        in_ctrl_qubits=ctrl_qubits,
        in_ctrl_values=ctrl_values_i1,
        adjoint=adjoint,
    ).results


#
# qubit unitary operation
#
@unitary_p.def_abstract_eval
def _unitary_abstract_eval(matrix, *qubits, qubits_len=0, ctrl_len=0, adjoint=False):
    for idx in range(qubits_len + ctrl_len):
        qubit = qubits[idx]
        assert isinstance(qubit, AbstractQbit)
    return (AbstractQbit(),) * (qubits_len + ctrl_len)


@unitary_p.def_impl
def _unitary_def_impl(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError()


def _unitary_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    matrix: ir.Value,
    *qubits_or_controlled: tuple,
    qubits_len=0,
    ctrl_len=0,
    adjoint=False,
):
    qubits = qubits_or_controlled[:qubits_len]
    ctrl_qubits = qubits_or_controlled[qubits_len : qubits_len + ctrl_len]
    ctrl_values = qubits_or_controlled[qubits_len + ctrl_len :]

    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    for q in qubits:
        assert ir.OpaqueType.isinstance(q.type)
        assert ir.OpaqueType(q.type).dialect_namespace == "quantum"
        assert ir.OpaqueType(q.type).data == "bit"

    matrix_type = matrix.type
    is_tensor = ir.RankedTensorType.isinstance(matrix_type)
    shape = ir.RankedTensorType(matrix_type).shape if is_tensor else None
    is_2d_tensor = len(shape) == 2 if is_tensor else False
    if not is_2d_tensor:
        raise TypeError("QubitUnitary must be a 2 dimensional tensor.")

    possibly_complex_type = ir.RankedTensorType(matrix_type).element_type
    is_complex = ir.ComplexType.isinstance(possibly_complex_type)
    is_f64_type = False

    if is_complex:
        complex_type = ir.ComplexType(possibly_complex_type)
        possibly_f64_type = complex_type.element_type
        is_f64_type = ir.F64Type.isinstance(possibly_f64_type)

    is_complex_f64_type = is_complex and is_f64_type
    if not is_complex_f64_type:
        f64_type = ir.F64Type.get()
        complex_f64_type = ir.ComplexType.get(f64_type)
        tensor_complex_f64_type = ir.RankedTensorType.get(shape, complex_f64_type)
        matrix = StableHLOConvertOp(tensor_complex_f64_type, matrix).result

    ctrl_values_i1 = []
    for v in ctrl_values:
        p = TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result
        ctrl_values_i1.append(p)

    return QubitUnitaryOp(
        out_qubits=[q.type for q in qubits],
        out_ctrl_qubits=[q.type for q in ctrl_qubits],
        matrix=matrix,
        in_qubits=qubits,
        in_ctrl_qubits=ctrl_qubits,
        in_ctrl_values=ctrl_values_i1,
        adjoint=adjoint,
    ).results


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


@pauli_rot_p.def_impl
def _pauli_rot_def_impl(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError()


# pylint: disable=unused-argument
def _pauli_rot_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *qubits_and_params: tuple,
    pauli_word=None,
    qubits_len=0,
    params_len=0,
    ctrl_len=0,
    adjoint=False,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    qubits = qubits_and_params[:qubits_len]
    params = qubits_and_params[qubits_len : qubits_len + params_len]
    ctrl_qubits = qubits_and_params[qubits_len + params_len : qubits_len + params_len + ctrl_len]
    ctrl_values = qubits_and_params[qubits_len + params_len + ctrl_len :]

    for q in qubits:
        assert ir.OpaqueType.isinstance(q.type)
        assert ir.OpaqueType(q.type).dialect_namespace == "quantum"
        assert ir.OpaqueType(q.type).data == "bit"

    assert params_len == 1 and params[0] is not None
    angle = params[0]
    angle = safe_cast_to_f64(angle, "PauliRot")
    angle = extract_scalar(angle, "PauliRot")
    assert ir.F64Type.isinstance(angle.type)
    assert pauli_word is not None

    pauli_word = ir.ArrayAttr.get([ir.StringAttr.get(p) for p in pauli_word])

    ctrl_values_i1 = []
    for v in ctrl_values:
        p = TensorExtractOp(ir.IntegerType.get_signless(1), v, []).result
        ctrl_values_i1.append(p)

    return PauliRotOp(
        out_qubits=[qubit.type for qubit in qubits],
        out_ctrl_qubits=[qubit.type for qubit in ctrl_qubits],
        angle=angle,
        pauli_product=pauli_word,
        adjoint=adjoint,
        in_qubits=qubits,
        in_ctrl_qubits=ctrl_qubits,
        in_ctrl_values=ctrl_values_i1,
    ).results


#
# pauli measure operation
#
@pauli_measure_p.def_abstract_eval
def _pauli_measure_abstract_eval(*qubits, pauli_word=None, qubits_len=0, adjoint=False):
    qubits = qubits[:qubits_len]
    assert all(isinstance(qubit, AbstractQbit) for qubit in qubits)
    # This corresponds to the measurement value and the qubits after the measurements
    return (core.ShapedArray((), bool),) + (AbstractQbit(),) * (qubits_len)


@pauli_measure_p.def_impl
def _pauli_measure_def_impl(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError()


def _pauli_measure_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *qubits: tuple,
    pauli_word=None,
    qubits_len=0,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    qubits = qubits[:qubits_len]
    for q in qubits:
        assert ir.OpaqueType.isinstance(q.type)
        assert ir.OpaqueType(q.type).dialect_namespace == "quantum"
        assert ir.OpaqueType(q.type).data == "bit"

    assert pauli_word is not None

    if not all(p in ["I", "X", "Y", "Z"] for p in pauli_word):
        raise ValueError("Only Pauli words consisting of 'I', 'X', 'Y', and 'Z' are allowed.")

    pauli_word = ir.ArrayAttr.get([ir.StringAttr.get(p) for p in pauli_word])

    result_type = ir.IntegerType.get_signless(1)

    ppm_results = PPMeasurementOp(
        out_qubits=[q.type for q in qubits],
        mres=result_type,
        pauli_product=pauli_word,
        in_qubits=qubits,
    ).results

    result, *out_qubits = ppm_results  # First element is the measurement result

    result_type = ir.RankedTensorType.get((), result.type)
    from_elements_op = FromElementsOp(result_type, result)

    return (from_elements_op.results[0],) + tuple(out_qubits)


#
# measure
#
@measure_p.def_abstract_eval
def _measure_abstract_eval(qubit, postselect: int = None):
    assert isinstance(qubit, AbstractQbit)
    return core.ShapedArray((), bool), qubit


@measure_p.def_impl
def _measure_def_impl(ctx, qubit, postselect: int = None):  # pragma: no cover
    raise NotImplementedError()


def _measure_lowering(jax_ctx: mlir.LoweringRuleContext, qubit: ir.Value, postselect: int = None):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qubit.type).data == "bit"

    # Prepare postselect attribute
    if postselect is not None:
        i32_type = ir.IntegerType.get_signless(32, ctx)
        postselect = ir.IntegerAttr.get(i32_type, postselect)

    result_type = ir.IntegerType.get_signless(1)

    result, new_qubit = MeasureOp(result_type, qubit.type, qubit, postselect=postselect).results

    result_from_elements_op = ir.RankedTensorType.get((), result.type)
    from_elements_op = FromElementsOp(result_from_elements_op, result)

    return (
        from_elements_op.results[0],
        new_qubit,
    )


#
# arbitrary-basis measurements
#
@measure_in_basis_p.def_abstract_eval
def _measure_in_basis_abstract_eval(
    angle: float, qubit: AbstractQbit, plane: MeasurementPlane, postselect: int = None
):
    assert isinstance(qubit, AbstractQbit)
    return core.ShapedArray((), bool), qubit


@measure_in_basis_p.def_impl
def _measure_in_basis_def_impl(
    ctx, angle: float, qubit: AbstractQbit, plane: MeasurementPlane, postselect: int = None
):  # pragma: no cover
    raise NotImplementedError()


def _measurement_plane_attribute(ctx, plane: MeasurementPlane):
    return ir.OpaqueAttr.get(
        "mbqc",
        ("measurement_plane " + MeasurementPlane(plane).name).encode("utf-8"),
        ir.NoneType.get(ctx),
        ctx,
    )


def _measure_in_basis_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    angle: float,
    qubit: ir.Value,
    plane: MeasurementPlane,
    postselect: int = None,
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qubit.type).data == "bit"

    angle = safe_cast_to_f64(angle, "angle")
    angle = extract_scalar(angle, "angle")

    assert ir.F64Type.isinstance(
        angle.type
    ), "Only scalar double parameters are allowed for quantum gates!"

    # Prepare postselect attribute
    if postselect is not None:
        i32_type = ir.IntegerType.get_signless(32, ctx)
        postselect = ir.IntegerAttr.get(i32_type, postselect)

    result_type = ir.IntegerType.get_signless(1)

    result, new_qubit = MeasureInBasisOp(
        result_type,
        qubit.type,
        qubit,
        plane=_measurement_plane_attribute(ctx, plane),
        angle=angle,
        postselect=postselect,
    ).results

    result_from_elements_op = ir.RankedTensorType.get((), result.type)
    from_elements_op = FromElementsOp(result_from_elements_op, result)

    return (
        from_elements_op.results[0],
        new_qubit,
    )


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


@compbasis_p.def_impl
def _compbasis_def_impl(ctx, *qubits_or_qreg, qreg_available):  # pragma: no cover
    raise NotImplementedError()


def _compbasis_lowering(
    jax_ctx: mlir.LoweringRuleContext, *qubits_or_qreg: tuple, qreg_available=False
):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.OpaqueType.get("quantum", "obs")

    if qreg_available:
        qreg = qubits_or_qreg[0]
        assert ir.OpaqueType.isinstance(qreg.type)
        assert ir.OpaqueType(qreg.type).dialect_namespace == "quantum"
        assert ir.OpaqueType(qreg.type).data == "reg"
        return ComputationalBasisOp(result_type, [], qreg=qreg).results

    else:
        qubits = qubits_or_qreg
        for qubit in qubits:
            assert ir.OpaqueType.isinstance(qubit.type)
            assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
            assert ir.OpaqueType(qubit.type).data == "bit"

        return ComputationalBasisOp(result_type, qubits).results


#
# named observable
#
@namedobs_p.def_impl
def _namedobs_def_impl(qubit, kind):  # pragma: no cover
    raise NotImplementedError()


@namedobs_p.def_abstract_eval
def _namedobs_abstract_eval(qubit, kind):
    assert isinstance(qubit, AbstractQbit)
    return AbstractObs()


def _named_obs_attribute(ctx, kind: str):
    return ir.OpaqueAttr.get(
        "quantum", ("named_observable " + kind).encode("utf-8"), ir.NoneType.get(ctx), ctx
    )


def _named_obs_lowering(jax_ctx: mlir.LoweringRuleContext, qubit: ir.Value, kind: str):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(qubit.type)
    assert ir.OpaqueType(qubit.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(qubit.type).data == "bit"

    obsId = _named_obs_attribute(ctx, kind)
    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return NamedObsOp(result_type, qubit, obsId).results


#
# hermitian observable
#
@hermitian_p.def_abstract_eval
def _hermitian_abstract_eval(matrix, *qubits):
    for q in qubits:
        assert isinstance(q, AbstractQbit)
    return AbstractObs()


@hermitian_p.def_impl
def _hermitian_def_impl(ctx, matrix, *qubits):  # pragma: no cover
    raise NotImplementedError()


def _hermitian_lowering(jax_ctx: mlir.LoweringRuleContext, matrix: ir.Value, *qubits: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return HermitianOp(result_type, matrix, qubits).results


#
# tensor observable
#
@tensorobs_p.def_impl
def _tensorobs_def_impl(ctx, *terms):  # pragma: no cover
    raise NotImplementedError()


@tensorobs_p.def_abstract_eval
def _tensorobs_abstract_eval(*terms):
    for o in terms:
        assert isinstance(o, AbstractObs)
    return AbstractObs()


def _tensor__obs_lowering(jax_ctx: mlir.LoweringRuleContext, *terms: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    result_type = ir.OpaqueType.get("quantum", "obs")

    return TensorOp(result_type, terms).results


#
# hamiltonian observable
#
@hamiltonian_p.def_abstract_eval
def _hamiltonian_abstract_eval(coeffs, *terms):
    for o in terms:
        assert isinstance(o, AbstractObs)
    return AbstractObs()


@hamiltonian_p.def_impl
def _hamiltonian_def_impl(ctx, coeffs, *terms):  # pragma: no cover
    raise NotImplementedError()


def _hamiltonian_lowering(jax_ctx: mlir.LoweringRuleContext, coeffs: ir.Value, *terms: tuple):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    coeffs = safe_cast_to_f64(coeffs, "Hamiltonian", "coefficient")

    result_type = ir.OpaqueType.get("quantum", "obs", ctx)

    return HamiltonianOp(result_type, coeffs, terms).results


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


@sample_p.def_impl
def _sample_def_impl(ctx, obs, *dynamic_shape, static_shape):  # pragma: no cover
    raise NotImplementedError()


def _sample_lowering(
    jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, *dynamic_shape, static_shape
):
    # result shape of sample op is (shots, number_of_qubits)
    # The static_shape argument contains the static dimensions, and None for dynamic dimensions

    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True
    f64_type = ir.F64Type.get()

    # Replace Nones in static_shape with dynamic mlir dimensions
    result_shape = tuple(ir.ShapedType.get_dynamic_size() if d is None else d for d in static_shape)
    result_type = ir.RankedTensorType.get(result_shape, f64_type)

    dynamic_shape = list(dynamic_shape)
    for i, dyn_dim in enumerate(dynamic_shape):
        dynamic_shape[i] = extract_scalar(dyn_dim, f"sample_dyn_dim_{i}")

    return SampleOp(result_type, obs, dynamic_shape=dynamic_shape).results


#
# counts measurement
#
def counts_staging_rule(jaxpr_trace, _src, obs, *dynamic_shape, static_shape):
    """
    The result shape of `counts_p` is (tensor<Nxf64>, tensor<Nxi64>)
    where N = 2**number_of_qubits.
    """

    shape = _merge_dyn_shape(static_shape, dynamic_shape)
    if obs.primitive is compbasis_p:
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


@counts_p.def_impl
def _counts_def_impl(ctx, obs, *dynamic_shape, static_shape):  # pragma: no cover
    raise NotImplementedError()


def _counts_lowering(
    jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, *dynamic_shape, static_shape
):
    # Note: result shape of counts op is (tensor<Nxf64>, tensor<Nxi64>)
    # where N = 2**number_of_qubits
    # This means even with dynamic shots, result shape is still static.
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    i64_type = ir.IntegerType.get_signless(64, ctx)
    f64_type = ir.F64Type.get()

    # Replace Nones in static_shape with dynamic mlir dimensions
    shape = tuple(ir.ShapedType.get_dynamic_size() if d is None else d for d in static_shape)
    eigvals_type = ir.RankedTensorType.get(shape, f64_type)
    counts_type = ir.RankedTensorType.get(shape, i64_type)

    if static_shape[0] is None:
        # dynamic num_qubits, pass the SSA value to the op
        dyn_shape = extract_scalar(dynamic_shape[0], "counts_size")
    else:
        # static num_qubits already in the shape, no need to pass another operand
        dyn_shape = None

    return CountsOp(eigvals_type, counts_type, obs, dynamic_shape=dyn_shape).results


#
# expval measurement
#
@expval_p.def_abstract_eval
def _expval_abstract_eval(obs, shape=None):
    assert isinstance(obs, AbstractObs)
    return core.ShapedArray((), jax.numpy.float64)


@expval_p.def_impl
def _expval_def_impl(ctx, obs, shape=None):  # pragma: no cover
    raise NotImplementedError()


def _expval_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shape=None):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(obs.type)
    assert ir.OpaqueType(obs.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(obs.type).data == "obs"

    result_type = ir.F64Type.get()

    mres = ExpvalOp(result_type, obs).result
    result_from_elements_op = ir.RankedTensorType.get((), result_type)
    from_elements_op = FromElementsOp(result_from_elements_op, mres)
    return from_elements_op.results


#
# var measurement
#
@var_p.def_abstract_eval
def _var_abstract_eval(obs, shape=None):
    assert isinstance(obs, AbstractObs)
    return core.ShapedArray((), jax.numpy.float64)


@var_p.def_impl
def _var_def_impl(ctx, obs, shape=None):  # pragma: no cover
    raise NotImplementedError()


def _var_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, shape=None):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    assert ir.OpaqueType.isinstance(obs.type)
    assert ir.OpaqueType(obs.type).dialect_namespace == "quantum"
    assert ir.OpaqueType(obs.type).data == "obs"

    result_type = ir.F64Type.get()

    mres = VarianceOp(result_type, obs).result
    result_from_elements_op = ir.RankedTensorType.get((), result_type)
    from_elements_op = FromElementsOp(result_from_elements_op, mres)
    return from_elements_op.results


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


@probs_p.def_impl
def _probs_def_impl(ctx, obs, *dynamic_shape, static_shape):  # pragma: no cover
    raise NotImplementedError()


def _probs_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, *dynamic_shape, static_shape):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    f64_type = ir.F64Type.get()

    # Replace Nones in static_shape with dynamic mlir dimensions
    result_shape = tuple(ir.ShapedType.get_dynamic_size() if d is None else d for d in static_shape)
    result_type = ir.RankedTensorType.get(result_shape, f64_type)

    if static_shape[0] is None:
        # dynamic sv_length, pass the SSA value to the op
        dyn_shape = extract_scalar(dynamic_shape[0], "probs_sv_length")
    else:
        # static sv_length already in the shape, no need to pass another operand
        dyn_shape = None
    return ProbsOp(result_type, obs, dynamic_shape=dyn_shape).results


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


@state_p.def_impl
def _state_def_impl(ctx, obs, *dynamic_shape, static_shape):  # pragma: no cover
    raise NotImplementedError()


def _state_lowering(jax_ctx: mlir.LoweringRuleContext, obs: ir.Value, *dynamic_shape, static_shape):
    ctx = jax_ctx.module_context.context
    ctx.allow_unregistered_dialects = True

    c64_type = ir.ComplexType.get(ir.F64Type.get())

    # Replace Nones in static_shape with dynamic mlir dimensions
    result_shape = tuple(ir.ShapedType.get_dynamic_size() if d is None else d for d in static_shape)
    result_type = ir.RankedTensorType.get(result_shape, c64_type)

    if static_shape[0] is None:
        # dynamic sv_length, pass the SSA value to the op
        dyn_shape = extract_scalar(dynamic_shape[0], "state_sv_length")
    else:
        # static sv_length already in the shape, no need to pass another operand
        dyn_shape = None
    return StateOp(result_type, obs, dynamic_shape=dyn_shape).results


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


@cond_p.def_impl
def _cond_def_impl(ctx, *preds_and_branch_args_plus_consts, branch_jaxprs):  # pragma: no cover
    raise NotImplementedError()


def _cond_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *preds_and_branch_args_plus_consts: tuple,
    branch_jaxprs: List[core.ClosedJaxpr],
    num_implicit_outputs: int,
):
    result_types = [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out]
    num_preds = len(branch_jaxprs) - 1
    preds = preds_and_branch_args_plus_consts[:num_preds]
    branch_args_plus_consts = preds_and_branch_args_plus_consts[num_preds:]
    flat_args_plus_consts = mlir.flatten_ir_values(branch_args_plus_consts)

    # recursively lower if-else chains to nested IfOps
    def emit_branches(preds, branch_jaxprs, ip):
        # ip is an MLIR InsertionPoint. This allows recursive calls to emit their Operations inside
        # the 'else' blocks of preceding IfOps.
        with ip:
            pred_extracted = TensorExtractOp(ir.IntegerType.get_signless(1), preds[0], []).result
            if_op_scf = IfOp(pred_extracted, result_types, hasElse=True)
            true_jaxpr = branch_jaxprs[0]
            if_block = if_op_scf.then_block

            # if block
            source_info_util.extend_name_stack("if")
            if_ctx = jax_ctx.replace(name_stack=jax_ctx.name_stack.extend("if"))
            with ir.InsertionPoint(if_block):
                # recursively generate the mlir for the if block
                (out, _) = mlir.jaxpr_subcomp(
                    if_ctx.module_context,
                    true_jaxpr.jaxpr,
                    if_ctx.name_stack,
                    mlir.TokenSet(),
                    [mlir.ir_constant(c) for c in true_jaxpr.consts],  # is never hit in our tests
                    *flat_args_plus_consts,
                    dim_var_values=jax_ctx.dim_var_values,
                    const_lowering=jax_ctx.const_lowering,
                )

                YieldOp(out)

            # else block
            source_info_util.extend_name_stack("else")
            else_ctx = jax_ctx.replace(name_stack=jax_ctx.name_stack.extend("else"))
            else_block = if_op_scf.else_block
            if len(preds) == 1:
                # Base case: reached the otherwise block
                otherwise_jaxpr = branch_jaxprs[-1]
                with ir.InsertionPoint(else_block):
                    (out, _) = mlir.jaxpr_subcomp(
                        else_ctx.module_context,
                        otherwise_jaxpr.jaxpr,
                        else_ctx.name_stack,
                        mlir.TokenSet(),
                        [mlir.ir_constants(c) for c in otherwise_jaxpr.consts],
                        *flat_args_plus_consts,
                        dim_var_values=jax_ctx.dim_var_values,
                        const_lowering=jax_ctx.const_lowering,
                    )

                    YieldOp(out)
            else:
                with ir.InsertionPoint(else_block) as else_ip:
                    child_if_op = emit_branches(preds[1:], branch_jaxprs[1:], else_ip)
                    YieldOp(child_if_op.results)
            return if_op_scf

    head_if_op = emit_branches(preds, branch_jaxprs, jax_ctx.module_context.ip.current)
    return head_if_op.results


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


@switch_p.def_impl
def _switch_def_impl(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError()


def _switch_lowering(
    jax_ctx,
    *index_and_cases_and_branch_args_plus_consts: tuple,
    branch_jaxprs: List[core.ClosedJaxpr],
    num_implicit_outputs: int,
):
    result_types = [mlir.aval_to_ir_types(outvar)[0] for outvar in branch_jaxprs[0].out_avals]

    index = index_and_cases_and_branch_args_plus_consts[0]
    # the last branch is default and does not have a case
    cases = index_and_cases_and_branch_args_plus_consts[1 : len(branch_jaxprs)]
    branch_args_plus_consts = index_and_cases_and_branch_args_plus_consts[len(branch_jaxprs) :]
    flat_args_plus_consts = mlir.flatten_ir_values(branch_args_plus_consts)

    index = _cast_to_index(index)

    # enumerate branches so that branch indices and "cases" line up properly
    cases = ir.DenseI64ArrayAttr.get(
        [case.owner.attributes["value"].get_splat_value().value for case in cases]
    )

    scf_switch_op = IndexSwitchOp(result_types, index, cases, len(branch_jaxprs) - 1)

    # construct switch branches
    for i in range(len(branch_jaxprs) - 1):
        with ir.InsertionPoint(scf_switch_op.caseRegions[i].blocks.append()):
            branch_ctx = jax_ctx.replace(name_stack=jax_ctx.name_stack.extend(f"branch {i}"))
            branch_jaxpr = branch_jaxprs[i]
            (out, _) = mlir.jaxpr_subcomp(
                branch_ctx.module_context,
                branch_jaxpr.jaxpr,
                branch_ctx.name_stack,
                mlir.TokenSet(),
                [mlir.ir_constant(const) for const in branch_jaxpr.consts],
                *flat_args_plus_consts,
                dim_var_values=jax_ctx.dim_var_values,
                const_lowering=jax_ctx.const_lowering,
            )

            YieldOp(out)

    with ir.InsertionPoint(scf_switch_op.defaultRegion.blocks.append()):
        branch_ctx = jax_ctx.replace(name_stack=jax_ctx.name_stack.extend("default branch"))
        branch_jaxpr = branch_jaxprs[-1]
        (out, _) = mlir.jaxpr_subcomp(
            branch_ctx.module_context,
            branch_jaxpr.jaxpr,
            branch_ctx.name_stack,
            mlir.TokenSet(),
            [mlir.ir_constant(const) for const in branch_jaxpr.consts],
            *flat_args_plus_consts,
            dim_var_values=jax_ctx.dim_var_values,
            const_lowering=jax_ctx.const_lowering,
        )

        YieldOp(out)

    return scf_switch_op.results


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


@while_p.def_impl
def _while_loop_def_impl(
    ctx,
    *iter_args_plus_consts,
    cond_jaxpr,
    body_jaxpr,
    cond_nconsts,
    body_nconsts,
    num_implicit_inputs,
    preserve_dimensions,
):  # pragma: no cover
    raise NotImplementedError()


def _while_loop_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *iter_args_plus_consts: tuple,
    cond_jaxpr: core.ClosedJaxpr,
    body_jaxpr: core.ClosedJaxpr,
    cond_nconsts: int,
    body_nconsts: int,
    num_implicit_inputs: int,
    preserve_dimensions: bool,
):
    loop_carry_types_plus_consts = [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_in]
    flat_args_plus_consts = mlir.flatten_ir_values(iter_args_plus_consts)
    assert [val.type for val in flat_args_plus_consts] == loop_carry_types_plus_consts

    # split the argument list into 3 separate groups
    # 1) the constants used in the condition function
    # 2) the constants used in the body function
    # 3) the normal arguments which are not constants
    cond_consts = flat_args_plus_consts[:cond_nconsts]
    body_consts = flat_args_plus_consts[cond_nconsts : cond_nconsts + body_nconsts]
    loop_args = flat_args_plus_consts[cond_nconsts + body_nconsts :]

    # remove const types from abstract parameter types list
    loop_carry_types = loop_carry_types_plus_consts[cond_nconsts + body_nconsts :]
    assert loop_carry_types == [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out], (
        loop_carry_types,
        [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out],
    )

    while_op_scf = WhileOp(loop_carry_types, loop_args)

    # cond block
    cond_block = while_op_scf.regions[0].blocks.append(*loop_carry_types)
    name_stack = jax_ctx.name_stack.extend("while")
    cond_ctx = jax_ctx.replace(name_stack=name_stack.extend("cond"))
    with ir.InsertionPoint(cond_block):
        cond_args = [cond_block.arguments[i] for i in range(len(loop_carry_types))]
        params = cond_consts + cond_args

        # recursively generate the mlir for the while cond
        ((pred,), _) = mlir.jaxpr_subcomp(
            cond_ctx.module_context,
            cond_jaxpr.jaxpr,
            cond_ctx.name_stack,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in cond_jaxpr.consts],
            *params,
            dim_var_values=jax_ctx.dim_var_values,
            const_lowering=jax_ctx.const_lowering,
        )

        pred_extracted = TensorExtractOp(ir.IntegerType.get_signless(1), pred, []).result
        ConditionOp(pred_extracted, cond_args)

    # body block
    body_block = while_op_scf.regions[1].blocks.append(*loop_carry_types)
    body_ctx = jax_ctx.replace(name_stack=name_stack.extend("body"))
    with ir.InsertionPoint(body_block):
        body_args = [body_block.arguments[i] for i in range(len(loop_carry_types))]
        params = body_consts + body_args

        # recursively generate the mlir for the while body
        out, _ = mlir.jaxpr_subcomp(
            body_ctx.module_context,
            body_jaxpr.jaxpr,
            body_ctx.name_stack,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in cond_jaxpr.consts],
            *params,
            dim_var_values=jax_ctx.dim_var_values,
            const_lowering=jax_ctx.const_lowering,
        )

        YieldOp(out)

    return while_op_scf.results


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


# pylint: disable=too-many-arguments
@for_p.def_impl
def _for_loop_def_impl(
    ctx,
    lower_bound,
    upper_bound,
    step,
    *iter_args_plus_consts,
    body_jaxpr,
    num_implicit_inputs=0,
    body_nconsts,
    preserve_dimensions,
):  # pragma: no cover
    raise NotImplementedError()


# pylint: disable=too-many-arguments
def _for_loop_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *iter_args_plus_consts: tuple,
    body_jaxpr: core.ClosedJaxpr,
    body_nconsts: int,
    apply_reverse_transform: bool,
    num_implicit_inputs: int,
    preserve_dimensions,
):
    body_consts = iter_args_plus_consts[:body_nconsts]
    body_implicits = iter_args_plus_consts[body_nconsts : body_nconsts + num_implicit_inputs]
    lower_bound = iter_args_plus_consts[body_nconsts + num_implicit_inputs + 0]
    upper_bound = iter_args_plus_consts[body_nconsts + num_implicit_inputs + 1]
    step = iter_args_plus_consts[body_nconsts + num_implicit_inputs + 2]
    loop_index = iter_args_plus_consts[body_nconsts + num_implicit_inputs + 3]
    loop_args = [*body_implicits, *iter_args_plus_consts[body_nconsts + num_implicit_inputs + 4 :]]

    loop_index_type = ir.RankedTensorType(loop_index.type).element_type

    all_param_types_plus_consts = [mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_in]
    assert [lower_bound.type, upper_bound.type, step.type] == all_param_types_plus_consts[
        body_nconsts + num_implicit_inputs : body_nconsts + num_implicit_inputs + 3
    ]
    assert [val.type for val in body_consts] == all_param_types_plus_consts[:body_nconsts]

    result_types = [v.type for v in loop_args]
    assert result_types == [
        mlir.aval_to_ir_types(a)[0] for a in jax_ctx.avals_out
    ], f"\n{result_types=} doesn't match \n{jax_ctx.avals_out=}"

    lower_bound, upper_bound, step = map(_cast_to_index, (lower_bound, upper_bound, step))

    if apply_reverse_transform:
        zero_np = np.array(0)
        one_np = np.array(1)
        zero_attr = ir.DenseIntElementsAttr.get(zero_np)
        one_attr = ir.DenseIntElementsAttr.get(one_np)
        zero_tensor = StableHLOConstantOp(zero_attr)
        one_tensor = StableHLOConstantOp(one_attr)
        ctx = jax_ctx.module_context.context
        i64_type = ir.IntegerType.get_signless(64, ctx)
        zero_i64 = TensorExtractOp(i64_type, zero_tensor, []).result
        one_i64 = TensorExtractOp(i64_type, one_tensor, []).result
        zero = IndexCastOp(ir.IndexType.get(), zero_i64).result
        one = IndexCastOp(ir.IndexType.get(), one_i64).result

        start_val, stop_val, step_val = lower_bound, upper_bound, step

        # Iterate from 0 to the number of iterations (ceil((stop - start) / step))
        distance = SubIOp(stop_val, start_val)
        num_iterations = CeilDivSIOp(distance.result, step_val)
        lower_bound, upper_bound, step = zero, num_iterations, one

    for_op_scf = ForOp(lower_bound, upper_bound, step, iter_args=loop_args)

    name_stack = jax_ctx.name_stack.extend("for")
    body_block = for_op_scf.body
    body_ctx = jax_ctx.replace(name_stack=name_stack.extend("body"))

    with ir.InsertionPoint(body_block):
        body_args = list(body_block.arguments)

        # Convert the index type iteration variable expected by MLIR to tensor<i64> expected by JAX.
        if apply_reverse_transform:
            # iv = start + normalized_iv * step
            body_args[0] = AddIOp(start_val, MulIOp(body_args[0], step_val))

        body_args[0] = IndexCastOp(loop_index_type, body_args[0]).result
        result_from_elements_op = ir.RankedTensorType.get((), loop_index_type)
        from_elements_op = FromElementsOp(result_from_elements_op, body_args[0])
        body_args[0] = from_elements_op.result

        # Re-order arguments in accordance with jax dynamic API convensions
        consts = body_consts
        loop_iter = body_args[0]
        implicit_args = body_args[1 : num_implicit_inputs + 1]
        explicit_args = body_args[num_implicit_inputs + 1 :]
        loop_params = (*consts, *implicit_args, loop_iter, *explicit_args)

        # Recursively generate the mlir for the loop body
        out, _ = mlir.jaxpr_subcomp(
            body_ctx.module_context,
            body_jaxpr.jaxpr,
            body_ctx.name_stack,
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in body_jaxpr.consts],
            *loop_params,
            dim_var_values=jax_ctx.dim_var_values,
            const_lowering=jax_ctx.const_lowering,
        )
        YieldOp(out)

    return for_op_scf.results


#
# assert
#
@assert_p.def_impl
def _assert_def_impl(ctx, assertion, error):  # pragma: no cover
    raise NotImplementedError()


@assert_p.def_abstract_eval
def _assert_abstract(assertion, error):
    return ()


def _assert_lowering(jax_ctx: mlir.LoweringRuleContext, assertion, error):
    assertion_mlir = TensorExtractOp(ir.IntegerType.get_signless(1), assertion, []).result
    AssertionOp(assertion=assertion_mlir, error=error)
    return ()


#
# state_prep
#
@set_state_p.def_impl
def set_state_impl(ctx, *qubits_or_params):  # pragma: no cover
    """Concrete evaluation"""
    raise NotImplementedError()


@set_state_p.def_abstract_eval
def set_state_abstract(*qubits_or_params):
    """Abstract evaluation"""
    length = len(qubits_or_params)
    qubits_length = length - 1
    return (AbstractQbit(),) * qubits_length


def _set_state_lowering(jax_ctx: mlir.LoweringRuleContext, *qubits_or_params):
    """Lowering of set state"""
    qubits_or_params = list(qubits_or_params)
    param = qubits_or_params.pop()
    qubits = qubits_or_params
    out_qubits = [qubit.type for qubit in qubits]
    return SetStateOp(out_qubits, param, qubits).results


#
# set_basis_state
#
@set_basis_state_p.def_impl
def set_basis_state_impl(ctx, *qubits_or_params):  # pragma: no cover
    """Concrete evaluation"""
    raise NotImplementedError()


@set_basis_state_p.def_abstract_eval
def set_basis_state_abstract(*qubits_or_params):
    """Abstract evaluation"""
    length = len(qubits_or_params)
    qubits_length = length - 1
    return (AbstractQbit(),) * qubits_length


def _set_basis_state_lowering(jax_ctx: mlir.LoweringRuleContext, *qubits_or_params):
    """Lowering of set basis state"""
    qubits_or_params = list(qubits_or_params)
    param = qubits_or_params.pop()
    qubits = qubits_or_params
    out_qubits = [qubit.type for qubit in qubits]
    return SetBasisStateOp(out_qubits, param, qubits).results


#
# adjoint
#
@adjoint_p.def_impl
def _adjoint_def_impl(ctx, *args, args_tree, jaxpr):  # pragma: no cover
    raise NotImplementedError()


@adjoint_p.def_abstract_eval
def _adjoint_abstract(*args, args_tree, jaxpr):
    return jaxpr.out_avals[-1:]


def _adjoint_lowering(
    jax_ctx: mlir.LoweringRuleContext,
    *args: Iterable[ir.Value],
    args_tree: PyTreeDef,
    jaxpr: core.ClosedJaxpr,
) -> ir.Value:
    """The JAX bind handler performing the Jaxpr -> MLIR adjoint lowering by taking the `jaxpr`
    expression to be lowered and all its already lowered arguments as MLIR value references. The Jax
    requires all the arguments to be passed as a single list of positionals, thus we pass indices of
    the argument groups. The handler returns the resulting MLIR Value."""

    # [1] - MLIR Value of constans, classical and quantum arguments [2] - JAXPR types of constans,
    # classical and quantum arguments [3] - Build a body of the adjoint operator. We pass constants
    # and classical arguments as-is, but substitute the quantum arguments with the arguments of the
    # block.

    ctx = jax_ctx.module_context.context
    consts, cargs, qargs = tree_unflatten(args_tree, args)  # [1]
    _, _, aqargs = tree_unflatten(args_tree, jax_ctx.avals_in)  # [2]

    assert len(qargs) == 1, "We currently expect exactly one quantum register argument"
    output_types = util.flatten(map(mlir.aval_to_ir_types, jax_ctx.avals_out))
    assert len(output_types) == 1 and output_types[0] == ir.OpaqueType.get(
        "quantum", "reg", ctx
    ), f"Expected a single result of quantum.register type, got: {output_types}"

    # Build an adjoint operation with a single-block region.
    op = AdjointOp(output_types[0], qargs[0])
    adjoint_block = op.regions[0].blocks.append(*[mlir.aval_to_ir_types(a)[0] for a in aqargs])
    with ir.InsertionPoint(adjoint_block):
        source_info_util.extend_name_stack("adjoint")
        out, _ = mlir.jaxpr_subcomp(
            jax_ctx.module_context,
            jaxpr.jaxpr,
            jax_ctx.name_stack.extend("adjoint"),
            mlir.TokenSet(),
            [mlir.ir_constants(c) for c in jaxpr.consts],
            *list(chain(consts, cargs, adjoint_block.arguments)),
            dim_var_values=jax_ctx.dim_var_values,
            const_lowering=jax_ctx.const_lowering,
        )

        QYieldOp([out[-1]])

    # Need to manually add adjoint lowering pass for PPR, since that pipeline is not
    # end-to-end yet.
    # TODO: remove this manual addition when PPR is end-to-end, or when PPR has its own
    # pipeline registered.

    if any(_op.name == "qec.ppr" for _op in adjoint_block.operations):

        def adjoint_pass_injector(_op: ir.Operation) -> ir.WalkResult:
            if _op.name == "transform.named_sequence":
                with ir.InsertionPoint.at_block_begin(_op.regions[0].blocks[0]):
                    ApplyRegisteredPassOp(
                        result=ir.OpaqueType.get("transform", 'op<"builtin.module">'),
                        target=_op.regions[0].blocks[0].arguments[0],  # just insert at beginning
                        pass_name="adjoint-lowering",
                        options={},
                        dynamic_options={},
                    )
                return ir.WalkResult.INTERRUPT
            return ir.WalkResult.ADVANCE

        op.parent.parent.walk(adjoint_pass_injector, walk_order=ir.WalkOrder.PRE_ORDER)

    return op.results


def safe_cast_to_f64(value, op, kind="parameter"):
    """Utility function to allow upcasting from integers and floats, while preventing downcasting
    from larger bitwidths or complex numbers."""
    assert ir.RankedTensorType.isinstance(value.type)

    baseType = ir.RankedTensorType(value.type).element_type
    if ir.ComplexType.isinstance(baseType) or (
        ir.FloatType.isinstance(baseType) and ir.FloatType(baseType).width > 64
    ):
        raise TypeError(
            f"Operator {op} expected a float64 {kind}, got {baseType}.\n"
            "If you didn't specify this operator directly, it may have come from the decomposition "
            "of a non-Unitary operator, such as an exponential with real exponent."
        )

    shape = ir.RankedTensorType(value.type).shape
    if not ir.F64Type.isinstance(baseType):
        targetBaseType = ir.F64Type.get()
        targetTensorType = ir.RankedTensorType.get(shape, targetBaseType)
        value = StableHLOConvertOp(targetTensorType, value).result

    return value


def _cast_to_index(p):
    p = TensorExtractOp(
        ir.RankedTensorType(p.type).element_type, p, []
    ).result  # tensor<i64> -> i64
    p = IndexCastOp(ir.IndexType.get(), p).result  # i64 -> index
    return p


def extract_scalar(value, op, kind="parameter"):
    """Utility function to extract real scalars from scalar tensors or one-element 1-D tensors."""
    assert ir.RankedTensorType.isinstance(value.type)

    baseType = ir.RankedTensorType(value.type).element_type
    shape = ir.RankedTensorType(value.type).shape
    if shape == []:
        value = TensorExtractOp(baseType, value, []).result
    elif shape == [1]:
        c0 = ConstantOp(ir.IndexType.get(), 0)
        value = TensorExtractOp(baseType, value, [c0]).result
    else:
        raise TypeError(f"Operator {op} expected a scalar {kind}, got tensor of shape {shape}")

    return value


# TODO: remove these patches after https://github.com/jax-ml/jax/pull/23886
def _sin_lowering2(ctx, x, accuracy):
    """Use hlo.sine lowering instead of the new sin lowering from jax 0.4.28"""
    return _nary_lower_hlo(hlo.sine, ctx, x, accuracy=accuracy)


def _cos_lowering2(ctx, x, accuracy):
    """Use hlo.cosine lowering instead of the new cosine lowering from jax 0.4.28"""
    return _nary_lower_hlo(hlo.cosine, ctx, x, accuracy=accuracy)


def subroutine_lowering(*args, **kwargs):
    """This is just a method that forwards arguments to _pjit_lowering

    Even though we could register the `pjit_p` lowering directly, this makes the code origin
    apparent in stack traces and similar use cases.
    """
    try:
        retval = _pjit_lowering(*args, **kwargs)
    except NotImplementedError as e:
        if "MLIR translation rule for primitive" in str(e):
            msg = (
                str(e)
                + """
                This error sometimes occurs when using quantum operations
                inside subroutines but calling them outside a qnode
            """
            )
            raise NotImplementedError(msg) from e
        raise e

    return retval


CUSTOM_LOWERING_RULES = (
    (zne_p, _zne_lowering),
    (device_init_p, _device_init_lowering),
    (device_release_p, _device_release_lowering),
    (qalloc_p, _qalloc_lowering),
    (qdealloc_p, _qdealloc_lowering),
    (qdealloc_qb_p, _qdealloc_qb_lowering),
    (qextract_p, _qextract_lowering),
    (qinsert_p, _qinsert_lowering),
    (qinst_p, _qinst_lowering),
    (num_qubits_p, _num_qubits_lowering),
    (gphase_p, _gphase_lowering),
    (unitary_p, _unitary_lowering),
    (pauli_rot_p, _pauli_rot_lowering),
    (pauli_measure_p, _pauli_measure_lowering),
    (measure_p, _measure_lowering),
    (compbasis_p, _compbasis_lowering),
    (namedobs_p, _named_obs_lowering),
    (hermitian_p, _hermitian_lowering),
    (tensorobs_p, _tensor__obs_lowering),
    (hamiltonian_p, _hamiltonian_lowering),
    (sample_p, _sample_lowering),
    (counts_p, _counts_lowering),
    (expval_p, _expval_lowering),
    (var_p, _var_lowering),
    (probs_p, _probs_lowering),
    (state_p, _state_lowering),
    (cond_p, _cond_lowering),
    (switch_p, _switch_lowering),
    (while_p, _while_loop_lowering),
    (for_p, _for_loop_lowering),
    (grad_p, _grad_lowering),
    (pl_jac_prim, _capture_grad_lowering),
    (func_p, _func_lowering),
    (jvp_p, _jvp_lowering),
    (vjp_p, _vjp_lowering),
    (adjoint_p, _adjoint_lowering),
    (print_p, _print_lowering),
    (assert_p, _assert_lowering),
    (python_callback_p, _python_callback_lowering),
    (value_and_grad_p, _value_and_grad_lowering),
    (set_state_p, _set_state_lowering),
    (set_basis_state_p, _set_basis_state_lowering),
    (sin_p, _sin_lowering2),
    (cos_p, _cos_lowering2),
    (quantum_kernel_p, _quantum_kernel_lowering),
    (quantum_subroutine_p, subroutine_lowering),
    (measure_in_basis_p, _measure_in_basis_lowering),
    (decomprule_p, _decomposition_rule_lowering),
)


def _scalar_abstractify(t):
    # pylint: disable=protected-access
    if t in {int, float, complex, bool} or isinstance(t, jax._src.numpy.scalar_types._ScalarMeta):
        return core.ShapedArray([], dtype=t, weak_type=True)
    raise TypeError(f"Argument type {t} is not a valid JAX type.")


pytype_aval_mappings[type] = _scalar_abstractify
pytype_aval_mappings[jax._src.numpy.scalar_types._ScalarMeta] = _scalar_abstractify
pytype_aval_mappings[ConcreteQbit] = lambda _: AbstractQbit()
pytype_aval_mappings[ConcreteQreg] = lambda _: AbstractQreg()
