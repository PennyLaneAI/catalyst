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

"""
This module implements a custom Jaxpr interpreter.

  https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html

This custom Jaxpr interpreter will interpret Jaxpr code generated from the tracing
of catalyst programs and it will interpret it in such a way that instead of interpreting
Catalyst operations, it will issue calls to cuda-quantum operations.

This effectively transforms a catalyst program into something that can generate cuda
quantum kernels.

This module also uses the CUDA-quantum API. Here is the reference:
  https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html
"""

import dataclasses
import json
from functools import wraps
from typing import List

import cudaq
import jax
from jax import numpy as jnp
from jax._src.util import safe_map

from catalyst.compilation_pipelines import QJIT_CUDA
from catalyst.compiler import CompileOptions
from catalyst.jax_primitives import (
    AbstractObs,
    compbasis_p,
    counts_p,
    qalloc_p,
    qdealloc_p,
    qdevice_p,
    qextract_p,
    qinsert_p,
    qinst_p,
    qmeasure_p,
    sample_p,
    state_p,
)
from catalyst.utils.jax_extras import remove_host_context

# We disable protected access in particular to avoid warnings with
# cudaq._pycuda.
# pylint: disable=protected-access
# And we disable unused-argument to avoid unused arguments in abstract_eval.
# Particularly those kwargs.
# pylint: disable=unused-argument
# This is for the interpreter loop.
# TODO: We can possibly remove the branches with a bit of indirection.
# pylint: disable=too-many-branches


class AbsCudaQState(jax.core.AbstractValue):
    "Abstract CUDA-quantum State."
    hash_value = hash("AbsCudaQState")

    def __eq__(self, other):
        return isinstance(other, AbsCudaQState)  # pragma: nocover

    def __hash__(self):
        return self.hash_value  # pragma: nocover


class CudaQState(cudaq.State):
    "Concrete CUDA-quantum state."
    aval = AbsCudaQState


class AbsCudaQbit(jax.core.AbstractValue):
    "Abstract CUDA-quantum qbit."
    hash_value = hash("AbsCudaQbit")

    def __eq__(self, other):
        return isinstance(other, AbsCudaQbit)  # pragma: nocover

    def __hash__(self):
        return self.hash_value  # pragma: nocover


class CudaQbit(cudaq._pycudaq.QuakeValue):
    "Concrete CUDA-quantum qbit."
    aval = AbsCudaQbit


class AbsCudaQReg(jax.core.AbstractValue):
    "Abstract CUDA-quantum quantum register."
    hash_value = hash("AbsCudaQReg")

    def __eq__(self, other):
        return isinstance(other, AbsCudaQReg)  # pragma: nocover

    def __hash__(self):
        return self.hash_value  # pragma: nocover


class CudaQReg(cudaq._pycudaq.QuakeValue):
    "Concrete CUDA-quantum quantum register."
    aval = AbsCudaQReg


class AbsCudaValue(jax.core.AbstractValue):
    "Abstract CUDA-quantum value."
    hash_value = hash("AbsCudaValue")

    def __eq__(self, other):
        return isinstance(other, AbsCudaValue)  # pragma: nocover

    def __hash__(self):
        return self.hash_value  # pragma: nocover


class CudaValue(cudaq._pycudaq.QuakeValue):
    "Concrete CUDA-quantum value."
    aval = AbsCudaValue


class AbsCudaKernel(jax.core.AbstractValue):
    "Abstract CUDA-quantum kernel."
    hash_value = hash("AbsCudaKernel")

    def __eq__(self, other):
        return isinstance(other, AbsCudaKernel)  # pragma: nocover

    def __hash__(self):
        return self.hash_value  # pragma: nocover


class CudaKernel(cudaq._pycudaq.QuakeValue):
    "Concrete CUDA-quantum kernel."
    aval = AbsCudaKernel


class AbsCudaSampleResult(jax.core.AbstractValue):
    "Abstract CUDA-quantum kernel."
    hash_value = hash("AbsCudaSampleResult")

    def __eq__(self, other):
        return isinstance(other, AbsCudaSampleResult)  # pragma: nocover

    def __hash__(self):
        return self.hash_value  # pragma: nocover


class CudaSampleResult(cudaq.SampleResult):
    "Concrete CUDA-quantum kernel."
    aval = AbsCudaSampleResult


jax.core.pytype_aval_mappings[CudaValue] = lambda x: x.aval
jax.core.pytype_aval_mappings[CudaQReg] = lambda x: x.aval
jax.core.pytype_aval_mappings[CudaQbit] = lambda x: x.aval
jax.core.pytype_aval_mappings[CudaSampleResult] = lambda x: x.aval
jax.core.pytype_aval_mappings[CudaQState] = lambda x: x.aval
jax.core.raise_to_shaped_mappings[AbsCudaValue] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaQReg] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaKernel] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaQbit] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaSampleResult] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaQState] = lambda aval, _: aval

# From the documentation
# https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html
# Let's do one by one...
# And also explicitly state which ones we are skipping for now.

# cudaq.make_kernel() -> cudaq.Kernel
cudaq_make_kernel_p = jax.core.Primitive("cudaq_make_kernel")


def cudaq_make_kernel():
    """Just a convenience function to bind the cudaq make kernel primitive."""
    return cudaq_make_kernel_p.bind()


@cudaq_make_kernel_p.def_impl
def cudaq_make_kernel_primitive_impl():
    """Concrete implementation of cudaq.make_kernel is just a call."""
    return cudaq.make_kernel()


@cudaq_make_kernel_p.def_abstract_eval
def cudaq_make_kernel_primitive_abs():
    """Abstract implementation of cudaq.make_kernel."""
    return AbsCudaKernel()


# cudaq.make_kernel(*args) -> tuple
# SKIP

# cudaq.from_state(kernel: cudaq.Kernel, qubits: cudaq.QuakeValue, state: numpy.ndarray[]) -> None
# SKIP

# cudaq.from_state(state: numpy.ndarray[]) -> cudaq.Kernel
# SKIP

# Allocate a single qubit and return a handle to it as a QuakeValue
# qalloc(self: cudaq.Kernel)                   -> cudaq.QuakeValue
# SKIP

# Allocate a register of qubits of size `qubit_count` and return a handle to them as a
# `QuakeValue`.
# qalloc(self: cudaq.Kernel, qubit_count: int) -> cudaq.QuakeValue

kernel_qalloc_p = jax.core.Primitive("kernel_qalloc")


def kernel_qalloc(kernel, size):
    """Convenience for binding."""
    return kernel_qalloc_p.bind(kernel, size)


@kernel_qalloc_p.def_impl
def kernel_qalloc_primitive_impl(kernel, size):
    """Concrete implementation."""
    return kernel.qalloc(size)


@kernel_qalloc_p.def_abstract_eval
def kernel_qalloc_primitive_abs(_kernel, _size):
    """Abstract evaluation."""
    return AbsCudaQReg()


qreg_getitem_p = jax.core.Primitive("qreg_getitem")


def qreg_getitem(qreg, idx):
    """Convenience for binding."""
    return qreg_getitem_p.bind(qreg, idx)


@qreg_getitem_p.def_impl
def qreg_getitem_primitive_impl(qreg, idx):
    """Concrete implementation."""
    return qreg[idx]


@qreg_getitem_p.def_abstract_eval
def qreg_getitem_primitive_abs(_qreg, _idx):
    """Abstract evaluation."""
    return AbsCudaQbit()


cudaq_getstate_p = jax.core.Primitive("cudaq_getstate")


def cudaq_getstate(kernel):
    """Convenience for binding."""
    return cudaq_getstate_p.bind(kernel)


@cudaq_getstate_p.def_impl
def cudaq_getstate_primitive_impl(kernel):
    """Concrete implementation."""
    return cudaq.get_state(kernel)


@cudaq_getstate_p.def_abstract_eval
def cudaq_getstate_primitive_abs(_kernel):
    """Abstract evaluation."""
    return AbsCudaQState()


# Allocate a register of qubits of size `qubit_count` (where `qubit_count` is an existing
# `QuakeValue`) and return a handle to them as a new `QuakeValue)
# SKIP

# Return the `Kernel` as a string in its MLIR representation using the Quke dialect.
# SKIP

# Just-In-Time (JIT) compile `self` (`Kernel`) and call the kernel function at the provided
# concrete arguments.
# __call__(self: cudaq.Kernel, *args) -> None
# SKIP

# Apply a x gate to the given target qubit or qubits
# x(self: cudaq.Kernel, target: cudaq.QuakeValue) -> None


def make_primitive_for_gate():
    """Just a function that wraps the functionality of making the kernel_inst primitive.

    This function will return:
      * gate_func: A convenience function for binding
      * kernel_gate_p: A JAX primitive for quantum gates.
    """
    kernel_gate_p = jax.core.Primitive("kernel_inst")
    kernel_gate_p.multiple_results = True

    def gate_func(kernel, *qubits_or_params, inst=None, qubits_len=-1):
        """Convenience.

        Quantum operations in CUDA-quantum return no values. But JAXPR expects return values.
        We can just say that multiple_results = True and return an empty tuple.
        """
        kernel_gate_p.bind(kernel, *qubits_or_params, inst=inst, qubits_len=qubits_len)
        return tuple()

    @kernel_gate_p.def_impl
    def gate_impl(kernel, *qubits_or_params, inst=None, qubits_len=-1):
        """Concrete implementation."""
        assert inst and qubits_len > 0
        method = getattr(cudaq.Kernel, inst)
        targets = qubits_or_params[:qubits_len]
        params = qubits_or_params[qubits_len:]
        method(kernel, *params, *targets)
        return tuple()

    @kernel_gate_p.def_abstract_eval
    def gate_abs(_kernel, *_qubits_or_params, inst=None, qubits_len=-1):
        """Abstract evaluation."""
        return tuple()

    return gate_func, kernel_gate_p


cuda_inst, cuda_inst_p = make_primitive_for_gate()


@dataclasses.dataclass(frozen=True)
class SideEffect(jax._src.effects.Effect):
    """Side effect token."""

    __str__ = lambda _: "SideEffect"  # pylint: disable=unnecessary-lambda-assignment


def make_primitive_for_m(gate: str):
    """A single function to make primitives for all measurement basis.

    Args:
      * gate (str): Either "x", "y", or "z" are valid values.
    Returns:
      * A function that binds a primitive.
      * the primitive itself.
    """

    assert gate in {"x", "y", "z"}
    gate = f"m{gate}"
    kernel_gate_p = jax.core.Primitive(f"kernel_{gate}")
    method = getattr(cudaq.Kernel, gate)

    def gate_func(kernel, target):
        """Convenience."""
        return kernel_gate_p.bind(kernel, target)

    @kernel_gate_p.def_impl
    def gate_impl(kernel, target):
        """Concrete implementation."""
        return method(kernel, target)

    @kernel_gate_p.def_effectful_abstract_eval
    def gate_abs(_kernel, _target):
        """Abstract evaluation with side-effect."""
        effects = set()
        effects.add(SideEffect)
        return AbsCudaValue(), effects

    return gate_func, kernel_gate_p


mx_call, mx_p = make_primitive_for_m("x")
my_call, my_p = make_primitive_for_m("y")
mz_call, mz_p = make_primitive_for_m("z")


cudaq_sample_p = jax.core.Primitive("cudaq_sample")
cudaq_counts_p = jax.core.Primitive("cudaq_counts")
cudaq_counts_p.multiple_results = True


def cudaq_sample(kernel, *args, shots_count=1000):
    """Convenience function for binding."""
    return cudaq_sample_p.bind(kernel, *args, shots_count=shots_count)


@cudaq_sample_p.def_impl
def cudaq_sample_impl(kernel, *args, shots_count=1000):
    """Concrete implementation of cudaq.sample.

    `cudaq.sample` returns an object which is a compressed version of what
    `qml.sample` returns as samples. Instead of returning an array with the observed
    population, `cudaq.sample` returns a dictionary where the keys are bitstrings and
    values are the frequency those bitstrings were observed.

    In a way `qml.count` is more similar to `cudaq.sample` than `qml.sample`.
    So, let's perform a little conversion here.
    """
    a_dict = cudaq.sample(kernel, *args, shots_count=shots_count)
    lls = [[k] * v for k, v in a_dict.items()]
    return [l for ls in lls for l in ls]


@cudaq_sample_p.def_abstract_eval
def cudaq_sample_abs(_kernel, *_args, shots_count=1000):
    """Abstract evaluation."""
    return AbsCudaSampleResult()


def cudaq_counts(kernel, *args, shape, shots_count=1000):
    """Convenience function for binding."""
    return cudaq_counts_p.bind(kernel, *args, shape=shape, shots_count=shots_count)


@cudaq_counts_p.def_impl
def cudaq_counts_impl(kernel, *args, shape=None, shots_count=1000):
    """Concrete implementation of counts.
    `cudaq.sample` returns an object which is a compressed version of what
    `qml.sample` returns as samples. Instead of returning an array with the observed
    population, `cudaq.sample` returns a dictionary where the keys are bitstrings and
    values are the frequency those bitstrings were observed.

    CUDA-quantum does not implement another function similar to `qml.counts`.
    The closest function is `cudaq.sample`.

    In Catalyst, `qml.counts` returns two arrays.
    The first array corresponds to a count from 0..shape
    denoting the integers that can be computed from the bitstrings.
    """

    strings = list(range(shape))
    res = {str(s): 0 for s in strings}

    a_dict = cudaq.sample(kernel, *args, shots_count=shots_count)
    # It looks like cuda uses a different endianness than catalyst.
    a_dict_decimal = {str(int(k[::-1], 2)): v for k, v in a_dict.items()}
    res.update(a_dict_decimal)

    # The integers are actually floats in Catalyst
    bitstrings, counts_items = zip(*[(float(k), v) for k, v in res.items()])

    return jnp.array(bitstrings), jnp.array(counts_items)


@cudaq_counts_p.def_abstract_eval
def cudaq_counts_abs(kernel, shape, shots_count=1000):
    """Abstract evaluation."""
    bitstrings = jax.core.ShapedArray([shape], jax.numpy.float64)
    counts_shape = jax.core.ShapedArray([shape], jax.numpy.int64)
    return bitstrings, counts_shape


# SKIP Async for the time being
# SKIP observe (spin_operator map is unclear at the moment)
# SKIP VQE
# Ignore everything else?

# There are no lowerings because we will not generate MLIR


def count(var: jax._src.core.Var):
    """Small function to get the identifier of a variable.

    In JAX, variables have a "count" attribute that is used to identify them.
    This corresponds to their name. 0 starts with the name a and as the count
    increases, the name also increases alphabetically.

    This function is safe and uses getattr because `eqn.invars` might return
    a `jax._src.core.Literal` which is not a `jax._src.core.Var` and has no count.
    """
    return getattr(var, "count", 0)


def counts(_vars: List[jax._src.core.Var]):
    """Get counts for all elements in _vars."""
    return map(count, _vars)


def invars(eqn: jax._src.core.JaxprEqn):
    """Make invars look like a function instead of an attribute.."""
    return eqn.invars


def outvars(eqn: jax._src.core.JaxprEqn):
    """Make outvars look like a function instead of an attribute.."""
    return eqn.outvars


def allvars(eqn: jax._src.core.JaxprEqn):
    """Create a list of all invars and outvars in an eqn."""
    return invars(eqn) + outvars(eqn)


def get_maximum_variable(jaxpr):
    """This function returns the maximum number of variables for the given jaxpr function.
    The count is an internal JAX detail that roughly corresponds to the variable name.

    We want the maximum name to avoid name collisions. I don't think anything happened
    when I didn't set max_count, but it is probably best to avoid collisions.
    """
    max_count = 1
    for eqn in jaxpr.eqns:
        max_count = max(0, max_count, *counts(allvars(eqn)))
    return max_count


def get_minimum_new_variable_count(jaxpr):
    """To make sure we avoid duplicating variable names, just find the maximum variable and add
    one."""
    return get_maximum_variable(jaxpr) + 1


def get_instruction(jaxpr, primitive):
    """We iterate through the JAXPR and find the first device instruction.

    A well formed JAXPR should only have a single device instruction for a quantum function.
    """
    for eqn in jaxpr.eqns:
        if eqn.primitive == primitive:
            return eqn

    return None  # pragma: nocover


class TranslatorContext:
    """This class keeps some state that is useful for interpreting an Catalyst and evaluating it in
    CUDA-quantum primitives.

    It has:
       * jaxpr: A reference to the program
       * env: Dict[jax.core._src.Var, AnyType] A map of variables to values.
       * inv_env: Dict[AnyType, jax.core._src.Var] A map from values to variables.
       * variable_map: Dict[jax.core._src.Var, jax.core._src.Var]: A map from variables
               in the old program to the new program.
       * count [int]: Keeps track of the last variable used.
    """

    def __init__(self, jaxpr, consts, *args):
        self.jaxpr = jaxpr
        self.env = {}
        self.variable_map = {}
        self.inv_env = {}
        safe_map(self.write, jaxpr.invars, args)
        safe_map(self.write, jaxpr.constvars, consts)
        self.count = get_minimum_new_variable_count(jaxpr)

    def read(self, var):
        """Read the value of variable var."""
        if isinstance(var, jax.core.Literal):
            return var.val
        if self.variable_map.get(var):
            var = self.variable_map[var]
        return self.env[var]

    def get_var_for_val(self, val):
        """Get the variable that holds value val."""
        return self.inv_env[val]

    def write(self, var, val):
        """var = val."""
        if self.variable_map.get(var):
            var = self.variable_map[var]
        self.inv_env[val] = var
        self.env[var] = val

    def replace(self, original, new):
        """Replace original variable with new variable."""
        self.variable_map[original] = new

    def get_new_count(self):
        """Increase count and return."""
        self.count += 1
        return self.count

    def new_variable(self, _type):
        """Convenience to get a new variable of a given type."""
        local_count = self.get_new_count()
        return jax._src.core.Var(local_count, "", _type)


def change_device_to_cuda_device(ctx):
    """Map Catalyst's qdevice_p primitive to its equivalent CUDA-quantum primitive
    as defined in this file.

    From here we get shots as well.
    """

    # The device here might also have some important information for
    # us. For example, the number of shots.

    qdevice_eqn = get_instruction(ctx.jaxpr, qdevice_p)

    # These parameters are stored in a json-like string.
    # We first convert the string to json.
    json_like_string = qdevice_eqn.params["rtd_kwargs"]
    json_like_string = json_like_string.replace("'", '"')
    json_like_string = json_like_string.replace("True", "true")
    json_string = json_like_string.replace("False", "false")

    # Finally, we load it
    parameters = json.loads(json_string)

    # Now we have the number of shots.
    # Shots are specified in PL at the very beginning, but in cuda
    # shots are not needed until the very end.
    # So, we will just return this variable
    # and it is the responsibility of the caller to propagate this information.
    shots = parameters.get("shots")

    kernel = cudaq_make_kernel()
    outvals = [kernel]
    outvariables = [ctx.new_variable(AbsCudaKernel())]
    safe_map(ctx.write, outvariables, outvals)
    return kernel, shots


def change_alloc_to_cuda_alloc(ctx, kernel):
    """Change Catalyst's qalloc_p primitive to a CUDA-quantum primitive.

    One difference between both primitives is that Catalyst's qalloc_p primitive
    does not require a parameter with a sub-program / kernel.

    CUDA-quantum does require a kernel as an operand.
    """

    # We know that there will only be one single qalloc instruction
    # in the generated code for each quantum node.
    # TODO: maybe add a debug mode that checks if the assumption above is true.
    # Do not check for all of them indiscriminately since it would take
    # too much compile time.
    eqn = get_instruction(ctx.jaxpr, qalloc_p)
    invals = safe_map(ctx.read, eqn.invars)
    # We know from the definition of qalloc_p
    # that there is only one operand
    # and the operand is the size of the register.
    size = invals[0]
    register = kernel_qalloc(kernel, size)
    outvals = [register]

    # We are creating a new variable that will replace the old
    # variable.
    outvariables = [ctx.new_variable(AbsCudaQReg())]
    safe_map(ctx.replace, eqn.outvars, outvariables)
    safe_map(ctx.write, eqn.outvars, outvals)
    return register


def change_register_getitem(ctx, eqn):
    """Change catalyst's qextract_p primitive to a CUDA-quantum primitive."""

    assert eqn.primitive == qextract_p
    invals = safe_map(ctx.read, eqn.invars)
    # We know from the definition of qextract_p
    # that it takes two operands.
    # The first one is the qreg and the second one is the
    # qubit index.
    # Because we have correctly mapped the replacement,
    # invals[0] should point to a correct cuda register.
    register = invals[0]
    idx = invals[1]
    cuda_qubit = qreg_getitem(register, idx)

    outvariables = [ctx.new_variable(AbsCudaQbit())]
    outvals = [cuda_qubit]

    safe_map(ctx.replace, eqn.outvars, outvariables)
    safe_map(ctx.write, eqn.outvars, outvals)


def change_register_setitem(ctx, eqn):
    """Set the correct post-conditions for CUDA-quantum when interpreting qinsert_p primitive

    This method is interesting because CUDA-quantum does not use value semantics for their qubits.
    This means that each qubit is transformed as a side effect of the operations. Since
    CUDA-quantum does not use value semantics for their qubits nor their quantum registers,
    it means that we must map from a value semantics program to a memory semantics program.

    How do we do that?
    We remove the extra SSA variables, but keep track of which variables refer to which qubits.
    """

    # There is no __setitem__ for a quake value.
    assert eqn.primitive == qinsert_p
    # We know from the definition of qinsert_p
    # that it takes three operands
    # * qreg_old
    # * idx
    # * qubit
    #
    # qinsert_p also returns a new qreg
    # which represents the new state of the previous
    # qreg after the insertion.
    # * qreg_new
    #
    # We are just going to map the new register to
    # the old register, and I think that will keep
    # the same semantics in place.
    invals = safe_map(ctx.read, eqn.invars)

    # Because invals has been replaced with the correct
    # variables, invals[0] now holds a reference to a cuda register
    old_register = invals[0]
    outvar = ctx.get_var_for_val(old_register)

    safe_map(ctx.replace, eqn.outvars, [outvar])
    safe_map(ctx.write, eqn.outvars, [old_register])


def change_instruction(ctx, eqn, kernel):
    """Change the instruction to one supported in CUDA-quantum."""

    assert eqn.primitive == qinst_p

    # This is the map of instruction names.
    from_catalyst_to_cuda = {
        "PauliX": "x",
        "PauliY": "y",
        "PauliZ": "z",
        "Hadamard": "h",
        "S": "s",
        "T": "t",
        "RX": "rx",
        "RY": "ry",
        "RZ": "rz",
    }

    # From the definition of qinst_p
    # Operands:
    # * qubits_or_params
    invals = safe_map(ctx.read, eqn.invars)
    qubits_or_params = invals

    # And two parameters:
    # * op=None
    # * qubits_len=-1
    params = eqn.params
    op = params["op"]
    cuda_inst_name = from_catalyst_to_cuda[op]
    qubits_len = params["qubits_len"]

    # Now, we can map to the correct op
    # For now just assume rx
    cuda_inst(kernel, *qubits_or_params, inst=cuda_inst_name, qubits_len=qubits_len)

    # Finally determine how many are qubits.
    qubits = qubits_or_params[:qubits_len]

    # Now that we did this, we need to remember
    # that handle_rx is not in SSA and will not return qubits
    # And so the eqn.outvars should be replaced with something.
    # Let's just replace them with the input values.
    safe_map(ctx.write, eqn.outvars, qubits)


def change_compbasis(ctx, eqn):
    """Compbasis in Catalyst essentially is the default observable."""
    assert eqn.primitive == compbasis_p

    # From compbasis_p's definition, its operands are:
    # * qubits
    qubits = safe_map(ctx.read, eqn.invars)

    # We dont have a use for compbasis yet.
    # So, the evaluation of it might as well just be the same.
    outvals = [AbstractObs(len(qubits), compbasis_p)]
    safe_map(ctx.write, eqn.outvars, outvals)


def change_get_state(ctx, eqn, kernel):
    """Change Catalyst's state_p to CUDA-quantum's state primitive."""
    assert eqn.primitive == state_p

    # From state_p's definition, its operands are:
    # * an observable
    # * a shape
    invals = safe_map(ctx.read, eqn.invars)
    # Just as state_p, we will only support compbasis.
    obs_catalyst = invals[0]
    if obs_catalyst.primitive is not compbasis_p:
        raise TypeError("state only supports computational basis")

    # We don't really care too much about the shape
    # It is only used for an assertion in Catalyst.
    # So, we will ignore it here.

    # To get a state in cuda we need a kernel
    # which does not flow from eqn.invars
    # so we get it from the parameter.
    cuda_state = cudaq_getstate(kernel)
    outvals = [cuda_state]
    outvariables = [ctx.new_variable(AbsCudaQState())]
    safe_map(ctx.replace, eqn.outvars, outvariables)
    safe_map(ctx.write, eqn.outvars, outvals)


def change_sample_or_counts(ctx, eqn, kernel):
    """Change Catalyst's sample_p or counts_p primitive to respective CUDA-quantum primitives."""

    is_sample = eqn.primitive == sample_p
    is_counts = eqn.primitive == counts_p
    is_valid = is_sample or is_counts
    assert is_valid

    # Sample and counts look the same in terms of
    # operands
    # * obs
    invals = safe_map(ctx.read, eqn.invars)

    # And parameters...
    # * shots
    # * shape
    params = eqn.params
    shots = params["shots"]

    # We will deal with compbasis in the same way as
    # when we deal with the state
    obs_catalyst = invals[0]
    # Technically, we can have other observables,
    # but at the moment in cuda quantum this is not yet implemented.
    if obs_catalyst.primitive is not compbasis_p:
        raise NotImplementedError("sample and counts only supports computational basis")

    if is_sample:
        shots_result = cudaq_sample(kernel, shots_count=shots)
        outvals = [shots_result]
        outvariables = [ctx.new_variable(AbsCudaSampleResult())]
        safe_map(ctx.replace, eqn.outvars, outvariables)
        safe_map(ctx.write, eqn.outvars, outvals)
    else:
        shape = 2**obs_catalyst.num_qubits
        outvals = cudaq_counts(kernel, shape=shape, shots_count=shots)
        bitstrings = jax.core.ShapedArray([shape], jax.numpy.float64)
        local_counts = jax.core.ShapedArray([shape], jax.numpy.int64)
        outvariables = [ctx.new_variable(bitstrings), ctx.new_variable(local_counts)]
        safe_map(ctx.replace, eqn.outvars, outvariables)
        safe_map(ctx.write, eqn.outvars, outvals)


def change_sample(ctx, eqn, kernel):
    """Convenience function. The name is the documentation."""
    return change_sample_or_counts(ctx, eqn, kernel)


def change_counts(ctx, eqn, kernel):
    """Convenience function. The name is the documentation."""
    return change_sample_or_counts(ctx, eqn, kernel)


def change_measure(ctx, eqn, kernel):
    """Change Catalyst's qmeasure_p to CUDA-quantum measure."""

    assert eqn.primitive == qmeasure_p

    # Operands to measure_p
    # *qubit
    invals = safe_map(ctx.read, eqn.invars)
    # Since we've already replaced it
    # this qubit refers to one in the cuda program.
    qubit = invals[0]

    # Cuda can measure in multiple basis.
    # Catalyst's measure op only measures in the Z basis.
    # So we map this measurement op to mz in cuda.
    result = mz_call(kernel, qubit)
    outvariables = [ctx.new_variable(AbsCudaValue()), ctx.new_variable(AbsCudaQbit())]
    outvals = [result, qubit]
    safe_map(ctx.replace, eqn.outvars, outvariables)
    safe_map(ctx.write, eqn.outvars, outvals)
    return result


def transform_jaxpr_to_cuda_jaxpr(jaxpr, consts, *args):
    """Implement a custom interpreter for Catalyst's JAXPR operands.
    Instead of interpreting Catalyst's JAXPR operands, we will execute
    CUDA-quantum equivalent instructions. As these operations are
    abstractly evaluated, they will be bound by JAX. The end result
    is that a new transform function will be traced."""

    ctx = TranslatorContext(jaxpr, consts, *args)
    # TODO: Do we need these shots?
    # It looks like measurement operations already come with their own shots value.
    kernel, _shots = change_device_to_cuda_device(ctx)
    # TODO: Do we need to keep track of this register.
    # It looks like other operations already come with the register variable.
    _register = change_alloc_to_cuda_alloc(ctx, kernel)
    measurement_set = set()

    # ignore set of instructions we don't care about.
    # because they have been handled before or they are just
    # not necessary in the CUDA-quantum API.
    ignore = {qdealloc_p, qdevice_p, qalloc_p}

    # Main interpreter loop.
    for eqn in jaxpr.eqns:
        if eqn.primitive == state_p:
            change_get_state(ctx, eqn, kernel)
        elif eqn.primitive == qextract_p:
            change_register_getitem(ctx, eqn)
        elif eqn.primitive == qinsert_p:
            change_register_setitem(ctx, eqn)
        elif eqn.primitive == qinst_p:
            change_instruction(ctx, eqn, kernel)
        elif eqn.primitive == compbasis_p:
            change_compbasis(ctx, eqn)
        elif eqn.primitive == sample_p:
            change_sample(ctx, eqn, kernel)
        elif eqn.primitive == counts_p:
            change_counts(ctx, eqn, kernel)
        elif eqn.primitive == qmeasure_p:
            # TODO: If we are returning the measurement
            # We must change it to sample with a single shot.

            # Otherwise, we will be returning a quake value
            # that is opaque and cannot be inspected for a value by the user.
            # For the time being, we can just add an exception if the return of
            # measurement is being returned directly.
            a_measurement = change_measure(ctx, eqn, kernel)
            # Keep track of measurements in a set.
            # This will be checked at the end to make sure that we do not return
            # a Quake value.
            measurement_set.add(a_measurement)
        elif eqn.primitive in ignore:
            continue

        # Do the normal interpretation...
        else:
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            ans = eqn.primitive.bind(*subfuns, *map(ctx.read, eqn.invars), **bind_params)
            if eqn.primitive.multiple_results:
                safe_map(ctx.write, eqn.outvars, ans)
            else:
                ctx.write(eqn.outvars[0], ans)

    retvals = safe_map(ctx.read, jaxpr.outvars)
    if set(retvals).issubset(measurement_set):
        raise NotImplementedError(
            "You cannot return measurements directly from a tape when compiling for cuda quantum."
        )
    return retvals


def catalyst_to_cuda(fun):
    """Wrapper function that takes a function that will be compiled to JAXPR.

    Args:
       fun: Function to be traced / converted into CUDA-quantum JAXPR.

    Returns:
       wrapped: A wrapped function that will do the tracing.
    """

    # TODO: kwargs?

    @wraps(fun)
    def wrapped(*args, **_kwargs):
        opts = CompileOptions()
        catalyst_jaxpr_with_host, _ = QJIT_CUDA(fun, opts).get_jaxpr(*args)
        catalyst_jaxpr = remove_host_context(catalyst_jaxpr_with_host)
        closed_jaxpr = jax._src.core.ClosedJaxpr(catalyst_jaxpr, catalyst_jaxpr.constvars)
        out = transform_jaxpr_to_cuda_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        return out

    return wrapped
