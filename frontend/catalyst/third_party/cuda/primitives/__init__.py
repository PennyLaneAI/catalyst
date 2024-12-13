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
This module implements JAXPR primitives for CUDA-quantum.
"""

from importlib.metadata import version

import cudaq
import jax
from jax import numpy as jnp

# We disable protected access in particular to avoid warnings with cudaq._pycuda.
# And we disable unused-argument to avoid unused arguments in abstract_eval, particularly kwargs.
# pylint: disable=protected-access,unused-argument,line-too-long


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


class AbsCudaSpinOperator(jax.core.AbstractValue):
    "Abstract CUDA-quantum spin operator."

    hash_value = hash("AbsCudaSpinOperator")

    def __eq__(self, other):
        return isinstance(other, AbsCudaSpinOperator)  # pragma: nocover

    def __hash__(self):
        return self.hash_value  # pragma: nocover


class CudaSpinOperator(cudaq.SpinOperator):
    "Concrete CUDA-quantum spin operator."
    aval = AbsCudaSpinOperator


class AbsCudaQObserveResult(jax.core.AbstractValue):
    "Abstract CUDA-quantum observe result."

    hash_value = hash("AbsCudaQObserveResult")

    def __eq__(self, other):
        return isinstance(other, AbsCudaQObserveResult)  # pragma: nocover

    def __hash__(self):
        return self.hash_value  # pragma: nocover


class CudaQObserveResult(cudaq.ObserveResult):
    "Concrete CUDA-quantum observe result."
    aval = AbsCudaQObserveResult


# From the documentation
# https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html
# Let's do one by one...
# And also explicitly state which ones we are skipping for now.

# cudaq.make_kernel() -> cudaq.Kernel
cudaq_make_kernel_p = jax.core.Primitive("cudaq_make_kernel")
cudaq_make_kernel_p.multiple_results = True


def cudaq_make_kernel(*args):
    """Just a convenience function to bind the cudaq make kernel primitive.
    From the documentation: https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.make_kernel

    The following types are supported as kernel arguments: int, float, list/List, cudaq.qubit,
    or cudaq.qreg.
    """
    return cudaq_make_kernel_p.bind(*args)


@cudaq_make_kernel_p.def_impl
def cudaq_make_kernel_primitive_impl(*args):
    """Concrete implementation of cudaq.make_kernel is just a call."""

    # This if statement is just to satisfy the multiple return values condition.
    # We need to return an iterable.
    if not args:
        return (cudaq.make_kernel(),)

    return cudaq.make_kernel(*args)


@cudaq_make_kernel_p.def_abstract_eval
def cudaq_make_kernel_primitive_abs(*args):
    """Abstract implementation of cudaq.make_kernel."""
    retvals = []
    retvals.append(AbsCudaKernel())
    retvals = retvals + list(args)
    return tuple(retvals)


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
    return jax.numpy.array(cudaq.get_state(kernel))


@cudaq_getstate_p.def_abstract_eval
def cudaq_getstate_primitive_abs(_kernel):  # pragma: nocover
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

    def gate_func(kernel, *qubits_or_params, inst=None, qubits_len=-1, static_params=None):
        """Convenience.

        Quantum operations in CUDA-quantum return no values. But JAXPR expects return values.
        We can just say that multiple_results = True and return an empty tuple.
        """
        kernel_gate_p.bind(kernel, *qubits_or_params, inst=inst, qubits_len=qubits_len, static_params=static_params)
        return tuple()

    @kernel_gate_p.def_impl
    def gate_impl(kernel, *qubits_or_params, inst=None, qubits_len=-1, static_params=None):
        """Concrete implementation."""
        assert inst and qubits_len > 0
        if static_params is None:
            static_params = []
        method = getattr(cudaq.Kernel, inst)
        targets = qubits_or_params[:qubits_len]
        params = qubits_or_params[qubits_len:]
        if not params:
            params = static_params
        method(kernel, *params, *targets)
        return tuple()

    @kernel_gate_p.def_abstract_eval
    def gate_abs(_kernel, *_qubits_or_params, inst=None, qubits_len=-1, static_params=None):
        """Abstract evaluation."""
        return tuple()

    return gate_func, kernel_gate_p


cuda_inst, _ = make_primitive_for_gate()


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
        # TODO(@erick-xanadu): investigate why nocover here.
        return method(kernel, target)  # pragma: nocover

    @kernel_gate_p.def_abstract_eval
    def gate_abs(_kernel, _target):
        """Abstract evaluation."""
        return AbsCudaValue()

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
    aggregate = []
    for bitstring, count in a_dict.items():
        bitarray = [int(bit) for bit in bitstring]
        for _ in range(count):
            aggregate.append(bitarray)

    return jax.numpy.array(aggregate)


@cudaq_sample_p.def_abstract_eval
def cudaq_sample_abs(_kernel, *_args, shots_count=1000):  # pragma: nocover
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

    res = {str(s): 0 for s in range(shape)}

    a_dict = cudaq.sample(kernel, *args, shots_count=shots_count)
    # It looks like cuda uses a different endianness than catalyst.
    a_dict_decimal = {str(int(k[::-1], 2)): v for k, v in a_dict.items()}
    res.update(a_dict_decimal)

    bitstrings, counts_items = zip(*[(int(k), v) for k, v in res.items()])

    return jnp.array(bitstrings), jnp.array(counts_items)


@cudaq_counts_p.def_abstract_eval
def cudaq_counts_abs(kernel, shape, shots_count=1000):  # pragma: nocover
    """Abstract evaluation."""
    bitstrings = jax.core.ShapedArray([shape], jax.numpy.float64)
    counts_shape = jax.core.ShapedArray([shape], jax.numpy.int64)
    return bitstrings, counts_shape


cudaq_spin_p = jax.core.Primitive("spin")


def cudaq_spin(target, kind: str):
    """Convenience function for spin."""
    assert kind in {"i", "x", "y", "z"}
    return cudaq_spin_p.bind(target, kind)


@cudaq_spin_p.def_impl
def cudaq_spin_impl(target, kind: str):
    """The spin operator."""
    method = getattr(cudaq.spin, kind)
    return method(target)


@cudaq_spin_p.def_abstract_eval
def cudaq_spin_abs(target, kind):  # pragma: nocover
    """Abstract spin operator."""
    return AbsCudaSpinOperator()


cudaq_observe_p = jax.core.Primitive("observe")


def cudaq_observe(kernel, spin_operator, shots_count=-1, noise_model=None):
    """Convenience wrapper around the primitive.

    From the documentation:

        https://nvidia.github.io/cuda-quantum/latest/api/languages/python_api.html#cudaq.observe

    Compute the expected value of the spin_operator with respect to the kernel.
    If the input spin_operator is a list of SpinOperator then compute the expected value of every
    operator in the list and return a list of results. If the kernel accepts arguments, it will
    be evaluated with respect to kernel(*arguments). Each argument in arguments provided can be a
    list or ndarray of arguments of the specified kernel argument type, and in this case, the
    observe functionality will be broadcasted over all argument sets and a list of observe_result
    instances will be returned. If both the input spin_operator and arguments are broadcast lists,
    a nested list of results over arguments then spin_operator will be returned.

    Some changes: spin_operator is only a single spin_operator instead of a list.
    TODO(@erick-xanadu): Can we generalize it to a list?

    The signature in the documentation also specifies an optional execution parameter but it is
    undocumented.

    The *args, have been ommitted, since we are building fully static circuits for now.
    """
    return cudaq_observe_p.bind(
        kernel, spin_operator, shots_count=shots_count, noise_model=noise_model
    )


@cudaq_observe_p.def_abstract_eval
def cudaq_observe_abs(kernel, spin_operator, shots_count=-1, noise_model=None):  # pragma: nocover
    """Abstract observe method."""
    return AbsCudaQObserveResult()


@cudaq_observe_p.def_impl
def cudaq_observe_impl(kernel, spin_operator, shots_count=-1, noise_model=None):
    """Concrete implementation."""
    return cudaq.observe(kernel, spin_operator, shots_count=shots_count, noise_model=noise_model)


cudaq_expectation_p = jax.core.Primitive("expectation")


def cudaq_expectation(observe_result):
    """Convenience."""
    return cudaq_expectation_p.bind(observe_result)


@cudaq_expectation_p.def_abstract_eval
def cudaq_expectation_abs(observe_result):  # pragma: nocover
    """Abstract."""
    return jax.core.ShapedArray([], float)


@cudaq_expectation_p.def_impl
def cudaq_expectation_impl(observe_result):
    """Concrete."""
    return observe_result.expectation()


cudaq_adjoint_p = jax.core.Primitive("cudaq_adjoint")
cudaq_adjoint_p.multiple_results = True


def cudaq_adjoint(kernel, target, *args):
    """Convenience."""
    cudaq_adjoint_p.bind(kernel, target, *args)
    return tuple()


@cudaq_adjoint_p.def_abstract_eval
def cudaq_adjoint_abs(kernel, target, *args):  # pragma: nocover
    """Abstract."""
    return tuple()


@cudaq_adjoint_p.def_impl
def cudaq_adjoint_impl(kernel, target, *args):
    """Concrete."""
    kernel.adjoint(target, *args)
    return tuple()


# SKIP Async for the time being
# SKIP VQE
# Ignore everything else?

# There are no lowerings because we will not generate MLIR
jax.core.pytype_aval_mappings[CudaValue] = lambda x: x.aval  # pragma: nocover
jax.core.pytype_aval_mappings[CudaQReg] = lambda x: x.aval  # pragma: nocover
jax.core.pytype_aval_mappings[CudaQbit] = lambda x: x.aval  # pragma: nocover
jax.core.pytype_aval_mappings[CudaSampleResult] = lambda x: x.aval  # pragma: nocover
jax.core.pytype_aval_mappings[CudaQState] = lambda x: x.aval  # pragma: nocover
jax.core.pytype_aval_mappings[CudaSpinOperator] = lambda x: x.aval  # pragma: nocover
jax.core.pytype_aval_mappings[CudaQObserveResult] = lambda x: x.aval  # pragma: nocover

# TODO: Investigate nocover in this ones.
jax.core.raise_to_shaped_mappings[AbsCudaValue] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaQReg] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaKernel] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaQbit] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaSampleResult] = lambda aval, _: aval  # pragma: nocover
jax.core.raise_to_shaped_mappings[AbsCudaQState] = lambda aval, _: aval
jax.core.raise_to_shaped_mappings[AbsCudaQObserveResult] = lambda aval, _: aval  # pragma: nocover
jax.core.raise_to_shaped_mappings[AbsCudaSpinOperator] = lambda aval, _: aval  # pragma: nocover
