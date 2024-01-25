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

import jax
import cudaq
from functools import wraps
from typing import List


class AbsCudaQState(jax.core.AbstractValue):
    hash_value = hash("AbsCudaQState")

    def __eq__(self, other):
        return isinstance(other, AbsCudaQState)

    def __hash__(self):
        return hash_value


class CudaQState(cudaq.State):
    aval = AbsCudaQState


class AbsCudaQbit(jax.core.AbstractValue):
    hash_value = hash("AbsCudaQbit")

    def __eq__(self, other):
        return isinstance(other, AbsCudaQbit)

    def __hash__(self):
        return self.hash_value


class CudaQbit(cudaq._pycudaq.QuakeValue):
    aval = AbsCudaQbit


class AbsCudaQReg(jax.core.AbstractValue):
    hash_value = hash("AbsCudaQReg")

    def __eq__(self, other):
        return isinstance(other, AbsCudaQReg)

    def __hash__(self):
        return self.hash_value


class CudaQReg(cudaq._pycudaq.QuakeValue):
    aval = AbsCudaQReg


class AbsCudaValue(jax.core.AbstractValue):
    hash_value = hash("AbsCudaValue")

    def __eq__(self, other):
        return isinstance(other, AbsCudaValue)

    def __hash__(self):
        return self.hash_value


class CudaValue(cudaq._pycudaq.QuakeValue):
    aval = AbsCudaValue


class AbsCudaKernel(jax.core.AbstractValue):
    hash_value = hash("AbsCudaKernel")

    def __eq__(self, other):
        return isinstance(other, AbsCudaKernel)

    def __hash__(self):
        return self.hash_value


class CudaKernel(cudaq._pycudaq.QuakeValue):
    aval = AbsCudaKernel


class AbsCudaSampleResult(jax.core.AbstractValue):
    hash_value = hash("AbsCudaSampleResult")

    def __eq__(self, other):
        return isinstance(other, AbsCudaSampleResult)

    def __hash__(self):
        return self.hash_value


class CudaSampleResult(cudaq.SampleResult):
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
    return cudaq_make_kernel_p.bind()


@cudaq_make_kernel_p.def_impl
def cudaq_make_kernel_primitive_impl():
    return cudaq.make_kernel()


@cudaq_make_kernel_p.def_abstract_eval
def cudaq_make_kernel_primitive_abs():
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

# Allocate a register of qubits of size `qubit_count` and return a handle to them as a `QuakeValue`.
# qalloc(self: cudaq.Kernel, qubit_count: int) -> cudaq.QuakeValue

kernel_qalloc_p = jax.core.Primitive("kernel_qalloc")


def kernel_qalloc(kernel, size):
    return kernel_qalloc_p.bind(kernel, size)


@kernel_qalloc_p.def_impl
def kernel_qalloc_primitive_impl(kernel, size):
    return kernel.qalloc(size)


@kernel_qalloc_p.def_abstract_eval
def kernel_qalloc_primitive_abs(kernel, size):
    return AbsCudaQReg()


qreg_getitem_p = jax.core.Primitive("qreg_getitem")


def qreg_getitem(qreg, idx):
    return qreg_getitem_p.bind(qreg, idx)


@qreg_getitem_p.def_impl
def qreg_getitem_primitive_impl(qreg, idx):
    return qreg[idx]


@qreg_getitem_p.def_abstract_eval
def qreg_getitem_primitive_abs(qreg, idx):
    return AbsCudaQbit()


cudaq_getstate_p = jax.core.Primitive("cudaq_getstate")


def cudaq_getstate(kernel):
    return cudaq_getstate_p.bind(kernel)


@cudaq_getstate_p.def_impl
def cudaq_getstate_primitive_impl(kernel):
    return cudaq.get_state(kernel)


@cudaq_getstate_p.def_abstract_eval
def cudaq_getstate_primitive_abs(kernel):
    return AbsCudaQState()


# Allocate a register of qubits of size `qubit_count` (where `qubit_count` is an existing `QuakeValue`) and return a
# handle to them as a new `QuakeValue)
# SKIP

# Return the `Kernel` as a string in its MLIR representation using the Quke dialect.
# SKIP

# Just-In-Time (JIT) compile `self` (`Kernel`) and call the kernel function at the provided concrete arguments.
# __call__(self: cudaq.Kernel, *args) -> None
# SKIP

# Apply a x gate to the given target qubit or qubits
# x(self: cudaq.Kernel, target: cudaq.QuakeValue) -> None


def make_primitive_for_gate():
    kernel_gate_p = jax.core.Primitive(f"kernel_inst")
    kernel_gate_p.multiple_results = True

    def gate_func(kernel, *qubits_or_params, inst=None, qubits_len=-1):
        kernel_gate_p.bind(kernel, *qubits_or_params, inst=inst, qubits_len=qubits_len)
        return tuple()

    @kernel_gate_p.def_impl
    def gate_impl(kernel, *qubits_or_params, inst=None, qubits_len=-1):
        assert inst
        method = getattr(cudaq.Kernel, inst)
        targets = qubits_or_params[:qubits_len]
        params = qubits_or_params[qubits_len:]
        method(kernel, *params, *targets)
        return tuple()

    @kernel_gate_p.def_abstract_eval
    def gate_abs(kernel, *qubits_or_params, inst=None, qubits_len=-1):
        return tuple()

    return gate_func, kernel_gate_p


cuda_inst, cuda_inst_p = make_primitive_for_gate()

import dataclasses


@dataclasses.dataclass(frozen=True)
class SideEffect(jax._src.effects.Effect):
    __str__ = lambda _: "SideEffect"


def make_primitive_for_m(gate: str):
    gate = f"m{gate}"
    kernel_gate_p = jax.core.Primitive(f"kernel_{gate}")
    method = getattr(cudaq.Kernel, gate)

    def gate_func(kernel, target):
        return kernel_gate_p.bind(kernel, target)

    @kernel_gate_p.def_impl
    def gate_impl(kernel, target):
        return method(kernel, target)

    @kernel_gate_p.def_effectful_abstract_eval
    def gate_abs(kernel, target):
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
    return cudaq_sample_p.bind(kernel, *args, shots_count=shots_count)


@cudaq_sample_p.def_impl
def cudaq_sample_impl(kernel, *args, shots_count=1000):
    # cudaq.sample returns an object which is a compressed version of what
    # qml.sample returns as samples. Instead of returning an array with the observed
    # population, cudaq.sample returns a dictionary where the keys are bitstrings and
    # values are the frequency those bitstrings were observed.

    # In a way qml.count is semantically equivalent to cudaq.sample
    # So, let's perform a little conversion here...
    a_dict = cudaq.sample(kernel, *args, shots_count=shots_count)
    lls = [[k] * v for k, v in a_dict.items()]
    return [l for ls in lls for l in ls]


@cudaq_sample_p.def_abstract_eval
def cudaq_sample_abs(kernel, *args, shots_count=1000):
    return AbsCudaSampleResult()


def cudaq_counts(kernel, *args, shape, shots_count=1000):
    return cudaq_counts_p.bind(kernel, *args, shape=shape, shots_count=shots_count)


@cudaq_counts_p.def_impl
def cudaq_counts_impl(kernel, *args, shape=None, shots_count=1000):
    # cudaq.sample returns an object which is a compressed version of what
    # qml.sample returns as samples. Instead of returning an array with the observed
    # population, cudaq.sample returns a dictionary where the keys are bitstrings and
    # values are the frequency those bitstrings were observed.

    # In Catalyst, counts returns two arrays.
    # The first array corresponds to a count from 0..shape
    # denoting the integers that can be computed from the bitstrings.
    import math
    from jax import numpy as jnp

    strings = [x for x in range(shape)]
    res = {str(s): 0 for s in strings}

    a_dict = cudaq.sample(kernel, *args, shots_count=shots_count)
    # It looks like cuda uses a different endianness than catalyst.
    a_dict_decimal = {str(int(k[::-1], 2)): v for k, v in a_dict.items()}
    res.update(a_dict_decimal)

    # The integers are actually floats in Catalyst
    bitstrings, counts = zip(*[(float(k), v) for k, v in res.items()])

    return jnp.array(bitstrings), jnp.array(counts)


@cudaq_counts_p.def_abstract_eval
def cudaq_counts_abs(kernel, shape, shots_count=1000):
    bitstrings = jax.core.ShapedArray([shape], jax.numpy.float64)
    counts = jax.core.ShapedArray([shape], jax.numpy.int64)
    return bitstrings, counts


# SKIP Async for the time being
# SKIP observe (spin_operator map is unclear at the moment)
# SKIP VQE
# Ignore everything else?

# There are no lowerings because we will not generate MLIR


def count(var: jax._src.core.Var):
    return getattr(var, "count", 0)


def counts(_vars: List[jax._src.core.Var]):
    return map(count, _vars)


def invars(eqn: jax._src.core.JaxprEqn):
    return eqn.invars


def outvars(eqn: jax._src.core.JaxprEqn):
    return eqn.outvars


def allvars(eqn: jax._src.core.JaxprEqn):
    return invars(eqn) + outvars(eqn)


def get_maximum_variable(jaxpr):
    """This function returns the maximum number of variables for the given jaxpr function.
    The count is an internal JAX detail that roughly corresponds to the variable name.

    """
    max_count = 1
    for eqn in jaxpr.eqns:
        max_count = max(0, max_count, *counts(allvars(eqn)))
    return max_count


def get_minimum_new_variable_count(jaxpr):
    """To make sure we avoid duplicating variable names, just find the maximum variable and add one."""
    return get_maximum_variable(jaxpr) + 1


def get_instruction(jaxpr, primitive):
    """We iterate through the JAXPR and find the first device instruction.

    A well formed JAXPR should only have a single device instruction for a quantum function.
    """
    for eqn in jaxpr.eqns:
        if eqn.primitive == primitive:
            return eqn


from jax._src.util import safe_map


class TranslatorContext:
    def __init__(self, jaxpr, consts, *args):
        self.jaxpr = jaxpr
        self.env = {}
        self.variable_map = {}
        self.inv_env = {}
        safe_map(self.write, jaxpr.invars, args)
        safe_map(self.write, jaxpr.constvars, consts)
        self.count = get_minimum_new_variable_count(jaxpr)

    def read(self, var):
        if type(var) is jax.core.Literal:
            return var.val
        if self.variable_map.get(var):
            var = self.variable_map[var]
        return self.env[var]

    def get_var_for_val(self, val):
        return self.inv_env[val]

    def write(self, var, val):
        if self.variable_map.get(var):
            var = self.variable_map[var]
        self.inv_env[val] = var
        self.env[var] = val

    def replace(self, original, new):
        self.variable_map[original] = new

    def get_new_count(self):
        self.count += 1
        return self.count

    def new_variable(self, _type):
        count = self.get_new_count()
        return jax._src.core.Var(count, "", _type)


def change_device_to_cuda_device(ctx):
    from catalyst.jax_primitives import qdevice_p
    import json

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
    outvars = [ctx.new_variable(AbsCudaKernel())]
    safe_map(ctx.write, outvars, outvals)
    return kernel, shots


def change_alloc_to_cuda_alloc(ctx, kernel):
    from catalyst.jax_primitives import qalloc_p

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
    outvars = [ctx.new_variable(AbsCudaQReg())]
    safe_map(ctx.replace, eqn.outvars, outvars)
    safe_map(ctx.write, eqn.outvars, outvals)
    return register


def change_register_getitem(ctx, eqn):
    from catalyst.jax_primitives import qextract_p

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

    outvars = [ctx.new_variable(AbsCudaQbit())]
    outvals = [cuda_qubit]

    safe_map(ctx.replace, eqn.outvars, outvars)
    safe_map(ctx.write, eqn.outvars, outvals)


def change_register_setitem(ctx, eqn):
    from catalyst.jax_primitives import qinsert_p

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
    from catalyst.jax_primitives import qinst_p

    assert eqn.primitive == qinst_p

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


def change_compbasis(ctx, eqn, kernel):
    from catalyst.jax_primitives import compbasis_p

    assert eqn.primitive == compbasis_p

    # From compbasis_p's definition, its operands are:
    # * qubits
    qubits = safe_map(ctx.read, eqn.invars)

    # We dont have a use for compbasis yet.
    # So, the evaluation of it might as well just be the same.
    from catalyst.jax_primitives import AbstractObs

    outvals = [AbstractObs(len(qubits), compbasis_p)]
    safe_map(ctx.write, eqn.outvars, outvals)
    # Note: Other observables won't be so easy...


def change_get_state(ctx, eqn, kernel):
    from catalyst.jax_primitives import state_p
    from catalyst.jax_primitives import compbasis_p

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
    outvars = [ctx.new_variable(AbsCudaQState())]
    safe_map(ctx.replace, eqn.outvars, outvars)
    safe_map(ctx.write, eqn.outvars, outvals)


def change_sample_or_counts(ctx, eqn, kernel):
    from catalyst.jax_primitives import sample_p, counts_p
    from catalyst.jax_primitives import compbasis_p

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
        outvars = [ctx.new_variable(AbsCudaSampleResult())]
        safe_map(ctx.replace, eqn.outvars, outvars)
        safe_map(ctx.write, eqn.outvars, outvals)
    else:
        shape = 2**obs_catalyst.num_qubits
        outvals = cudaq_counts(kernel, shape=shape, shots_count=shots)
        bitstrings = jax.core.ShapedArray([shape], jax.numpy.float64)
        counts = jax.core.ShapedArray([shape], jax.numpy.int64)
        outvars = [ctx.new_variable(bitstrings), ctx.new_variable(counts)]
        safe_map(ctx.replace, eqn.outvars, outvars)
        safe_map(ctx.write, eqn.outvars, outvals)


def change_sample(ctx, eqn, kernel):
    return change_sample_or_counts(ctx, eqn, kernel)


def change_counts(ctx, eqn, kernel):
    return change_sample_or_counts(ctx, eqn, kernel)


def change_measure(ctx, eqn, kernel):
    from catalyst.jax_primitives import qmeasure_p

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
    outvars = [ctx.new_variable(AbsCudaValue()), ctx.new_variable(AbsCudaQbit())]
    outvals = [result, qubit]
    safe_map(ctx.replace, eqn.outvars, outvars)
    safe_map(ctx.write, eqn.outvars, outvals)


def transform_jaxpr_to_cuda_jaxpr(jaxpr, consts, *args):
    from jax._src.util import safe_map
    from catalyst.jax_primitives import (
        qdevice_p,
        qalloc_p,
        state_p,
        qextract_p,
        qinst_p,
        compbasis_p,
        qinsert_p,
        qdealloc_p,
        sample_p,
        counts_p,
        qmeasure_p,
    )

    ignore = {qdealloc_p, qdevice_p, qalloc_p}

    ctx = TranslatorContext(jaxpr, consts, *args)
    kernel, shots = change_device_to_cuda_device(ctx)
    register = change_alloc_to_cuda_alloc(ctx, kernel)

    for idx, eqn in enumerate(jaxpr.eqns):
        if eqn.primitive == state_p:
            change_get_state(ctx, eqn, kernel)
        elif eqn.primitive == qextract_p:
            change_register_getitem(ctx, eqn)
        elif eqn.primitive == qinsert_p:
            change_register_setitem(ctx, eqn)
        elif eqn.primitive == qinst_p:
            change_instruction(ctx, eqn, kernel)
        elif eqn.primitive == compbasis_p:
            change_compbasis(ctx, eqn, kernel)
        elif eqn.primitive == sample_p:
            change_sample(ctx, eqn, kernel)
        elif eqn.primitive == counts_p:
            change_counts(ctx, eqn, kernel)
        elif eqn.primitive == qmeasure_p:
            change_measure(ctx, eqn, kernel)
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

    return safe_map(ctx.read, jaxpr.outvars)


# So where is this function going to be called?
# fun has to be a function that returns jaxpr.
# Normally, we would trace it via `make_jaxpr`.
# But with Catalyst, we are no longer going through that route.
def catalyst_to_cuda(fun):
    """This will likely become what lives in @qjit when cuda-quantum is selected as compiler."""

    from catalyst.compilation_pipelines import qjit_catalyst, QJIT_CUDA
    from catalyst.compiler import CompileOptions
    from catalyst.utils.jax_extras import remove_host_context

    @wraps(fun)
    def wrapped(*args, **kwargs):
        opts = CompileOptions()
        catalyst_jaxpr_with_host = QJIT_CUDA(fun, opts).get_jaxpr(*args)
        catalyst_jaxpr = remove_host_context(catalyst_jaxpr_with_host)
        closed_jaxpr = jax._src.core.ClosedJaxpr(catalyst_jaxpr, catalyst_jaxpr.constvars)
        out = transform_jaxpr_to_cuda_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        return out

    return wrapped
