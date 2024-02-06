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

import functools
import json
import operator
from functools import reduce, wraps
from typing import List

import cudaq
import jax
import pennylane as qml
from jax._src.util import safe_map
from jax.tree_util import tree_unflatten

from catalyst.jax_primitives import (
    AbstractObs,
    adjoint_p,
    compbasis_p,
    cond_p,
    counts_p,
    expval_p,
    for_p,
    func_p,
    grad_p,
    hamiltonian_p,
    hermitian_p,
    jvp_p,
    namedobs_p,
    print_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qdevice_p,
    qextract_p,
    qinsert_p,
    qinst_p,
    qmeasure_p,
    qunitary_p,
    sample_p,
    state_p,
    tensorobs_p,
    var_p,
    vjp_p,
    while_p,
    zne_p,
)
from catalyst.jax_tracer import trace_to_jaxpr
from catalyst.pennylane_extensions import QFunc
from catalyst.utils.jax_extras import remove_host_context
from catalyst.utils.patching import Patcher

from .primitives import (
    AbsCudaKernel,
    AbsCudaQbit,
    AbsCudaQReg,
    AbsCudaQState,
    AbsCudaSampleResult,
    AbsCudaSpinOperator,
    AbsCudaValue,
    cuda_inst,
    cudaq_adjoint,
    cudaq_counts,
    cudaq_expectation,
    cudaq_getstate,
    cudaq_make_kernel,
    cudaq_observe,
    cudaq_sample,
    cudaq_spin,
    kernel_qalloc,
    mz_call,
    qreg_getitem,
)

# We disable protected access in particular to avoid warnings with
# cudaq._pycuda.
# pylint: disable=protected-access
# This is for the interpreter loop.
# TODO: We can possibly remove the branches with a bit of indirection.
# pylint: disable=too-many-branches
# We also disable line-too-long because of a URL
# pylint: disable=line-too-long


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


class InterpreterContext:
    """This class keeps some state that is useful for interpreting an Catalyst and evaluating it in
    CUDA-quantum primitives.

    It has:
       * jaxpr: A reference to the program
       * env: Dict[jax.core._src.Var, AnyType] A map of variables to values.
       * variable_map: Dict[jax.core._src.Var, jax.core._src.Var]: A map from variables
               in the old program to the new program.
       * count [int]: Keeps track of the last variable used.
       * kernel: The main trace of the computation. There can be subkernels (e.g., adjoint)
    """

    def __init__(self, jaxpr, consts, *args, kernel=None):
        self.jaxpr = jaxpr
        self.env = {}
        self.variable_map = {}
        self.qubit_to_wire_map = {}
        safe_map(self.write, jaxpr.invars, args)
        safe_map(self.write, jaxpr.constvars, consts)
        self.count = get_minimum_new_variable_count(jaxpr)
        if kernel is None:
            # TODO: Do we need these shots?
            # It looks like measurement operations already come with their own shots value.
            self.kernel, _shots = change_device_to_cuda_device(self)
            change_alloc_to_cuda_alloc(self, self.kernel)
        else:
            # This is equivalent to passing a qreg into a function.
            # The kernel comes from outside the scope.
            self.kernel = kernel

    def set_qubit_to_wire(self, idx, qubit):
        """Keep track of which wire this qubit variable is."""
        self.qubit_to_wire_map[qubit] = idx

    def get_wire_for_qubit(self, qubit):
        """Get the wire for this qubit."""
        return self.qubit_to_wire_map[qubit]

    def read(self, var):
        """Read the value of variable var."""
        if isinstance(var, jax.core.Literal):
            return var.val
        if self.variable_map.get(var):
            var = self.variable_map[var]
        return self.env[var]

    def write(self, var, val):
        """var = val."""
        if self.variable_map.get(var):
            var = self.variable_map[var]
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

    # cudaq_make_kernel returns a multiple values depending on the arguments.
    # Here it is returning a single value wrapped in a list.
    kernel = cudaq_make_kernel()
    outvariables = [ctx.new_variable(AbsCudaKernel())]
    safe_map(ctx.write, outvariables, kernel)

    return kernel[0], shots


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
    ctx.set_qubit_to_wire(idx, cuda_qubit)

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
    # This is old we need to do.
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
    # This is an assert as opposed to raising an error,
    # because there appears to be no code that can lead to this
    # outcome.
    assert obs_catalyst.primitive == compbasis_p

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
    # This is an assert as opposed to raising an error,
    # because there appears to be no code that can lead to this
    # outcome.
    assert obs_catalyst.primitive == compbasis_p

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


def change_expval(ctx, eqn, kernel):
    """Change Catalyst's expval to CUDA-quantum equivalent."""
    assert eqn.primitive == expval_p

    # Operands to expval_p
    # * obs: Observables
    invals = safe_map(ctx.read, eqn.invars)
    obs = invals[0]

    # Params:
    # * shots: Shots
    shots = eqn.params["shots"]
    shots = shots if shots is not None else -1

    # To obtain expval, we first obtain an observe object.
    observe_results = cudaq_observe(kernel, obs, shots)
    # And then we call expectation on that object.
    result = cudaq_expectation(observe_results)
    outvariables = [ctx.new_variable(jax.core.ShapedArray([], float))]
    outvals = [result]

    safe_map(ctx.replace, eqn.outvars, outvariables)
    safe_map(ctx.write, eqn.outvars, outvals)


def change_namedobs(ctx, eqn):
    """Change named observable to CUDA-quantum equivalent."""
    assert eqn.primitive == namedobs_p

    # Operands to expval_p
    # * qubit
    invals = safe_map(ctx.read, eqn.invars)
    qubit = invals[0]

    # Since CUDA doesn't use SSA for qubits.
    # We now need to get an integer from this qubit.
    # This will be the target to cudaq_spin....

    # How do we get the wire from the SSA value?
    # We need to keep a map.
    # Where for every qubit variable, it maps to an integer.
    # This is not too hard.
    idx = ctx.get_wire_for_qubit(qubit)

    # Parameters
    # * kind
    kind = eqn.params["kind"]
    catalyst_cuda_map = {"PauliZ": "z", "PauliX": "x", "PauliY": "y"}

    # This is an assert because it is guaranteed by Catalyst.
    assert kind in catalyst_cuda_map

    cuda_name = catalyst_cuda_map[kind]
    outvals = [cudaq_spin(idx, cuda_name)]
    outvariables = [ctx.new_variable(AbsCudaSpinOperator())]

    safe_map(ctx.replace, eqn.outvars, outvariables)
    safe_map(ctx.write, eqn.outvars, outvals)


def change_hamiltonian(ctx, eqn):
    """Change catalyst hamiltonian to an equivalent expression in CUDA."""
    assert eqn.primitive == hamiltonian_p

    invals = safe_map(ctx.read, eqn.invars)
    coeffs = invals[0]
    terms = invals[1:]

    hamiltonian = reduce(operator.add, map(operator.mul, coeffs, terms))

    outvariables = [ctx.new_variable(AbsCudaSpinOperator())]
    safe_map(ctx.replace, eqn.outvars, outvariables)
    safe_map(ctx.write, eqn.outvars, [hamiltonian])


def change_adjoint(ctx, eqn):
    """Change Catalyst adjoint to an equivalent expression in CUDA."""
    assert eqn.primitive == adjoint_p

    # This is the quantum register.
    invals = safe_map(ctx.read, eqn.invars)
    register = invals[0]

    # We might need this for pytrees.
    _args_tree = eqn.params["args_tree"]
    nested_jaxpr = eqn.params["jaxpr"]

    # So, first we need to make a new kernel.
    # And we need to pass a register type as an argument
    # we are working on as an argument.
    # cudaq.qreg is essentially an abstract type in cudaq
    kernel_to_adjoint, abstract_qreg = cudaq_make_kernel(cudaq.qreg)

    # We need a new interpreter with a new kernel
    # And the parameter is abstract_qreg.
    interpreter = InterpreterContext(
        nested_jaxpr.jaxpr,
        nested_jaxpr.literals,
        abstract_qreg,
        kernel=kernel_to_adjoint,
    )
    # retval would be the abstract_qreg we passed as an argument here.
    # We always pass the qreg and return the qreg.
    # TODO: Do we support returning any other values?
    _retval = interpret_impl(interpreter, nested_jaxpr.jaxpr)
    cudaq_adjoint(ctx.kernel, kernel_to_adjoint, register)

    # Now that we have captured all of the adjoint...
    # Let's actually create an operation with it.
    safe_map(ctx.write, eqn.outvars, [register])


def interpret_impl(ctx, jaxpr):
    """Implement a custom interpreter for Catalyst's JAXPR operands.
    Instead of interpreting Catalyst's JAXPR operands, we will execute
    CUDA-quantum equivalent instructions. As these operations are
    abstractly evaluated, they will be bound by JAX. The end result
    is that a new transform function will be traced. If they are concretely evaluated,
    then it outputs a result instead of a trace.

    Args:
       ctx: An interpreter with the correct context
       jaxpr: Jaxpr to be interpreted

    Returns:
       Either concrete values or a trace that corresponds to the computed values.
    """
    measurement_set = set()

    # ignore set of instructions we don't care about.
    # because they have been handled before or they are just
    # not necessary in the CUDA-quantum API.
    ignore = {qdealloc_p, qdevice_p, qalloc_p}

    # TODO: Let's implement all of these
    unimplemented = {
        zne_p,
        qunitary_p,
        compbasis_p,
        hermitian_p,
        tensorobs_p,
        var_p,
        probs_p,
        cond_p,
        while_p,
        for_p,
        grad_p,
        func_p,
        jvp_p,
        vjp_p,
        adjoint_p,
        print_p,
    }

    # Main interpreter loop.
    # Note the absense of loops here and branches.
    # Normally in an interpreter loop, we would have
    # while(True):
    for eqn in jaxpr.eqns:
        # Leave this as a "switch-dispatch" and do not yet change to something like direct-call threading.
        # https://www.cs.toronto.edu/~matz/dissertation/matzDissertation-latex2html/node6.html
        # Yes, python does not have a switch statement.
        # Why not change to direct-call threading equivalent in Python yet?
        # I want to have an interpreter object that has this state.
        # But first, I want to implement all the requiremed primitives.
        if eqn.primitive == state_p:
            change_get_state(ctx, eqn, ctx.kernel)
        elif eqn.primitive == qextract_p:
            change_register_getitem(ctx, eqn)
        elif eqn.primitive == qinsert_p:
            change_register_setitem(ctx, eqn)
        elif eqn.primitive == qinst_p:
            change_instruction(ctx, eqn, ctx.kernel)
        elif eqn.primitive == compbasis_p:
            change_compbasis(ctx, eqn)
        elif eqn.primitive == sample_p:
            change_sample(ctx, eqn, ctx.kernel)
        elif eqn.primitive == counts_p:
            change_counts(ctx, eqn, ctx.kernel)
        elif eqn.primitive == qmeasure_p:
            # TODO: If we are returning the measurement
            # We must change it to sample with a single shot.

            # Otherwise, we will be returning a quake value
            # that is opaque and cannot be inspected for a value by the user.
            # For the time being, we can just add an exception if the return of
            # measurement is being returned directly.
            a_measurement = change_measure(ctx, eqn, ctx.kernel)
            # Keep track of measurements in a set.
            # This will be checked at the end to make sure that we do not return
            # a Quake value.
            measurement_set.add(a_measurement)
        elif eqn.primitive == expval_p:
            change_expval(ctx, eqn, ctx.kernel)
        elif eqn.primitive == namedobs_p:
            change_namedobs(ctx, eqn)
        elif eqn.primitive == hamiltonian_p:
            change_hamiltonian(ctx, eqn)
        elif eqn.primitive == adjoint_p:
            change_adjoint(ctx, eqn)
        elif eqn.primitive in unimplemented:
            msg = f"{eqn.primitive} is not yet implemented in Catalyst's CUDA-Quantum support."
            raise NotImplementedError(msg)
        elif eqn.primitive in ignore:
            continue

        # Do the normal interpretation...
        else:
            # This little scope was based on eval_jaxpr's implmentation:
            #    https://github.com/google/jax/blob/16636f9c97414d0c5195c6fd47227756d4754095/jax/_src/core.py#L507-L518
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            ans = eqn.primitive.bind(*subfuns, *map(ctx.read, eqn.invars), **bind_params)
            if eqn.primitive.multiple_results:  # pragma: nocover
                safe_map(ctx.write, eqn.outvars, ans)
            else:
                ctx.write(eqn.outvars[0], ans)

    retvals = safe_map(ctx.read, jaxpr.outvars)
    if set(retvals).issubset(measurement_set):
        raise NotImplementedError(
            "You cannot return measurements directly from a tape when compiling for cuda quantum."
        )
    return retvals


class QJIT_CUDA:
    """Class representing a just-in-time compiled hybrid quantum-classical function.

    .. note::

        ``QJIT_CUDA`` objects are created by the :func:`~.qjit` decorator. Please see
        the :func:`~.qjit` documentation for more details.

    Args:
        fn (Callable): the quantum or classical function
    """

    def __init__(self, fn):
        self.user_function = fn
        functools.update_wrapper(self, fn)

    def get_jaxpr(self, *args):
        """Trace :func:`~.user_function`

        Args:
            *args: either the concrete values to be passed as arguments to ``fn`` or abstract values

        Returns:
            an MLIR module
        """

        with Patcher(
            (qml.QNode, "__call__", QFunc.__call__),
        ):
            func = self.user_function
            abs_axes = {}
            static_args = None
            # _jaxpr and _out_tree are used in Catalyst but at the moment
            # they have no use here in CUDA.
            # We could also pass abstract arguments here in *args
            # the same way we do so in Catalyst.
            # But I think that is redundant now given make_jaxpr2
            _jaxpr, jaxpr2, _out_type2, out_tree = trace_to_jaxpr(
                func, static_args, abs_axes, *args
            )

        # TODO(@erick-xanadu):
        # What about static_args?
        # We could return _out_type2 as well
        return jaxpr2, out_tree


def interpret(fun):
    """Wrapper function that takes a function that will be interpretted with
    the semantics in interpret_impl.

    Args:
       fun: Function to be interpretted.

    Returns:
       wrapped: A wrapped function that will do the tracing.
    """

    # TODO(@erick-xanadu): kwargs?

    @wraps(fun)
    def wrapped(*args, **_kwargs):
        # QJIT_CUDA(fun).get_jaxpr
        # will return the JAXPR of the function fun.
        # However, notice that *args are still concrete.
        # This means that when we interpret the JAXPR, it will be a concrete interpretation.
        # If we wanted an abstract interpretation, we could pass the abstract values that
        # correspond to the return value of trace_to_jaxpr out_type2.
        # If we did that, we could cache the result JAXPR from this function
        # and evaluate it each time the function is called.
        catalyst_jaxpr_with_host, out_tree = QJIT_CUDA(fun).get_jaxpr(*args)

        # We need to keep track of the consts...
        consts = catalyst_jaxpr_with_host.consts
        catalyst_jaxpr = remove_host_context(catalyst_jaxpr_with_host)
        closed_jaxpr = jax._src.core.ClosedJaxpr(catalyst_jaxpr, catalyst_jaxpr.constvars)

        # Because they become args...
        args = list(args) + consts
        args = tuple(args)
        ctx = InterpreterContext(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        out = interpret_impl(ctx, closed_jaxpr.jaxpr)

        out = tree_unflatten(out_tree, out)
        return out

    return wrapped
