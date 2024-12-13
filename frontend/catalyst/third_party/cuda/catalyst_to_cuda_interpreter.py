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
from typing import Hashable

import cudaq
import jax
import pennylane as qml
from jax.tree_util import tree_unflatten

from catalyst.device import BackendInfo, QJITDevice
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
    quantum_kernel_p,
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
from catalyst.qfunc import QFunc
from catalyst.utils.exceptions import CompileError
from catalyst.utils.patching import Patcher

from .primitives import (
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


def remove_host_context(jaxpr):
    """This function will remove the host context.

    It is expected that callers to this function do not use the host context
    and instead forward the arguments directly to the qnode.
    This is useful for example in the proof-of-concept for cuda integration.

    Later iterations **might** have a host context. This is something not currently planned.

    The host context is the wrapper function that calls a qnode. When applying a `@qjit`
    decorator immediately on a qnode like:

    ```python
    @qjit
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def identity(x):
      return x
    ```

    Notice that in the JAXPR (and all subsequent IRs), we have this wrapper context that only
    performs a function call.

    { lambda ; a:i64[]. let
      b:i64[] = func[
        call_jaxpr={ lambda ; c:i64[]. let
             = qdevice[
              rtd_kwargs={'shots': 0, 'mcmc': False}
              rtd_lib=path/to/runtime...
              rtd_name=LightningSimulator
            ]
            d:AbstractQreg() = qalloc 1
             = qdealloc d
          in (c,) }
        fn=<QNode: wires=1, device='lightning.qubit', interface='auto', diff_method='best'>
      ] a
    in (b,) }
    """
    is_one_equation = len(jaxpr.jaxpr.eqns) == 1
    prim = jaxpr.jaxpr.eqns[0].primitive
    is_single_equation_call = prim in {func_p, quantum_kernel_p}
    is_valid = is_one_equation and is_single_equation_call
    if is_valid:
        return jaxpr.jaxpr.eqns[0][3]["call_jaxpr"]
    raise CompileError("Cannot translate tapes with context.")


def _map(f, *collections):
    """Eager implementation of map."""
    return list(map(f, *collections))


def get_instruction(jaxpr, primitive):
    """We iterate through the JAXPR and find the first device instruction.

    A well formed JAXPR should only have a single device instruction for a quantum function.
    """
    return next((eqn for eqn in jaxpr.eqns if eqn.primitive == primitive), None)  # pragma: no cover


class InterpreterContext:
    """This class keeps some state that is useful for interpreting Catalyst's JAXPR and evaluating
    it in CUDA-quantum primitives.

    It has:
       * jaxpr: A reference to the program
       * env: Dict[jax.core._src.Var, AnyType] A map of variables to values.
       * variable_map: Dict[jax.core._src.Var, jax.core._src.Var]: A map from variables
               in the old program to the new program.
       * count [int]: Keeps track of the last variable used.
       * kernel: The main trace of the computation. There can be subkernels (e.g., adjoint)
    """

    def __init__(self, jaxpr, consts, *args, kernel=None):
        self.count = 0
        self.jaxpr = jaxpr
        self.env = {}
        self.variable_map = {}
        self.qubit_to_wire_map = {}
        _map(self.write, jaxpr.invars, args)
        _map(self.write, jaxpr.constvars, consts)
        self.measurements = set()
        if kernel is None:
            # TODO: Do we need these shots?
            # It looks like measurement operations already come with their own shots value.
            self.kernel, _shots = change_device_to_cuda_device(self)
            change_alloc_to_cuda_alloc(self, self.kernel)
        else:
            # This is equivalent to passing a qreg into a function.
            # The kernel comes from outside the scope.
            self.kernel = kernel

    def add_measurement(self, m):
        """Keep track of measurements such that we don't return them.
        Measurements are quake values that are opaque to the user.
        They are essentially compile time references to potential values.
        """
        self.measurements.add(m)

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

    def new_variable(self):
        """Convenience to get a new variable of a given type.

        These are not JAX variables. This can be any hashable.
        We are using integers as a small level of indirection.
        Each new variable is a new integer.

        This is pretty much what JAX does since each variable gets
        assigned a count and a type. But it is also what anyone else does.
        (Like LLVM/MLIR %0, %1, ...)

        TODO(@erick-xanadu): I think the type information
        would also be good to have here, but it is unnecessary at the
        moment.
        """
        return self.get_new_count()


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

    device_name = qdevice_eqn.params.get("rtd_name")

    if not cudaq.has_target(device_name):
        msg = f"Unavailable target {device_name}."  # pragma: no cover
        raise ValueError(msg)

    cudaq_target = cudaq.get_target(device_name)
    cudaq.set_target(cudaq_target)

    # cudaq_make_kernel returns a multiple values depending on the arguments.
    # Here it is returning a single value wrapped in a list.
    kernel = cudaq_make_kernel()
    outvariables = [ctx.new_variable()]
    _map(ctx.write, outvariables, kernel)

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
    invals = _map(ctx.read, eqn.invars)
    # We know from the definition of qalloc_p
    # that there is only one operand
    # and the operand is the size of the register.
    size = invals[0]
    register = kernel_qalloc(kernel, size)
    outvals = [register]

    # We are creating a new variable that will replace the old
    # variable.
    outvariables = [ctx.new_variable()]
    _map(ctx.replace, eqn.outvars, outvariables)
    _map(ctx.write, eqn.outvars, outvals)
    return register


def change_register_getitem(ctx, eqn):
    """Change catalyst's qextract_p primitive to a CUDA-quantum primitive."""

    assert eqn.primitive == qextract_p
    invals = _map(ctx.read, eqn.invars)
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

    outvariables = [ctx.new_variable()]
    outvals = [cuda_qubit]

    _map(ctx.replace, eqn.outvars, outvariables)
    _map(ctx.write, eqn.outvars, outvals)


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
    invals = _map(ctx.read, eqn.invars)
    # Because invals has been replaced with the correct
    # variables, invals[0] now holds a reference to a cuda register
    old_register = invals[0]
    # This is old we need to do.
    _map(ctx.write, eqn.outvars, [old_register])


def change_instruction(ctx, eqn):
    """Change the instruction to one supported in CUDA-quantum."""

    assert eqn.primitive == qinst_p

    # This is the map of instruction names.
    from_catalyst_to_cuda = {
        "CNOT": "cx",
        "CY": "cy",
        "CZ": "cz",
        "CRX": "crx",
        "CRY": "cry",
        "CRZ": "crz",
        "PauliX": "x",
        "PauliY": "y",
        "PauliZ": "z",
        "Hadamard": "h",
        "S": "s",
        "T": "t",
        "RX": "rx",
        "RY": "ry",
        "RZ": "rz",
        "SWAP": "swap",
        "CSWAP": "cswap",
        # Other instructions that are missing:
        # ch
        # sdg
        # tdg
        # cs
        # ct
        # r1
    }

    # From the definition of qinst_p
    # Operands:
    # * qubits_or_params
    invals = _map(ctx.read, eqn.invars)
    qubits_or_params = invals

    # And two parameters:
    # * op=None
    # * qubits_len=-1
    params = eqn.params
    op = params["op"]
    cuda_inst_name = from_catalyst_to_cuda[op]
    qubits_len = params["qubits_len"]
    static_params = params.get("static_params")

    # Now, we can map to the correct op
    # For now just assume rx
    cuda_inst(ctx.kernel, *qubits_or_params, inst=cuda_inst_name, qubits_len=qubits_len, static_params=static_params)

    # Finally determine how many are qubits.
    qubits = qubits_or_params[:qubits_len]

    # Now that we did this, we need to remember
    # that handle_rx is not in SSA and will not return qubits
    # And so the eqn.outvars should be replaced with something.
    # Let's just replace them with the input values.
    _map(ctx.write, eqn.outvars, qubits)


def change_compbasis(ctx, eqn):
    """Compbasis in Catalyst essentially is the default observable."""
    assert eqn.primitive == compbasis_p

    # From compbasis_p's definition, its operands are:
    # * qubits
    qubits = _map(ctx.read, eqn.invars)

    # We dont have a use for compbasis yet.
    # So, the evaluation of it might as well just be the same.
    outvals = [AbstractObs(len(qubits), compbasis_p)]
    _map(ctx.write, eqn.outvars, outvals)


def change_get_state(ctx, eqn):
    """Change Catalyst's state_p to CUDA-quantum's state primitive."""
    assert eqn.primitive == state_p

    # From state_p's definition, its operands are:
    # * an observable
    # * a shape
    invals = _map(ctx.read, eqn.invars)
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
    cuda_state = cudaq_getstate(ctx.kernel)
    outvals = [cuda_state]
    outvariables = [ctx.new_variable()]
    _map(ctx.replace, eqn.outvars, outvariables)
    _map(ctx.write, eqn.outvars, outvals)


def change_sample_or_counts(ctx, eqn):
    """Change Catalyst's sample_p or counts_p primitive to respective CUDA-quantum primitives."""

    is_sample = eqn.primitive == sample_p
    is_counts = eqn.primitive == counts_p
    is_valid = is_sample or is_counts
    assert is_valid

    # Sample and counts look the same in terms of
    # operands
    # * obs
    invals = _map(ctx.read, eqn.invars)

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
        shots_result = cudaq_sample(ctx.kernel, shots_count=shots)
        outvals = [shots_result]
        outvariables = [ctx.new_variable()]
        _map(ctx.replace, eqn.outvars, outvariables)
        _map(ctx.write, eqn.outvars, outvals)
    else:
        shape = 2**obs_catalyst.num_qubits
        outvals = cudaq_counts(ctx.kernel, shape=shape, shots_count=shots)
        outvariables = [ctx.new_variable(), ctx.new_variable()]
        _map(ctx.replace, eqn.outvars, outvariables)
        _map(ctx.write, eqn.outvars, outvals)


def change_sample(ctx, eqn):
    """Convenience function. The name is the documentation."""
    return change_sample_or_counts(ctx, eqn)


def change_counts(ctx, eqn):
    """Convenience function. The name is the documentation."""
    return change_sample_or_counts(ctx, eqn)


def change_measure(ctx, eqn):
    """Change Catalyst's qmeasure_p to CUDA-quantum measure."""

    assert eqn.primitive == qmeasure_p

    # Operands to measure_p
    # *qubit
    invals = _map(ctx.read, eqn.invars)
    # Since we've already replaced it
    # this qubit refers to one in the cuda program.
    qubit = invals[0]

    # Cuda can measure in multiple basis.
    # Catalyst's measure op only measures in the Z basis.
    # So we map this measurement op to mz in cuda.
    result = mz_call(ctx.kernel, qubit)
    outvariables = [ctx.new_variable(), ctx.new_variable()]
    outvals = [result, qubit]
    _map(ctx.replace, eqn.outvars, outvariables)
    _map(ctx.write, eqn.outvars, outvals)
    # TODO: If we are returning the measurement
    # We must change it to sample with a single shot.

    # Otherwise, we will be returning a quake value
    # that is opaque and cannot be inspected for a value by the user.
    # For the time being, we can just add an exception if the return of
    # measurement is being returned directly.
    ctx.add_measurement(result)


def change_expval(ctx, eqn):
    """Change Catalyst's expval to CUDA-quantum equivalent."""
    assert eqn.primitive == expval_p

    # Operands to expval_p
    # * obs: Observables
    invals = _map(ctx.read, eqn.invars)
    obs = invals[0]

    # To obtain expval, we first obtain an observe object.
    observe_results = cudaq_observe(ctx.kernel, obs)
    # And then we call expectation on that object.
    result = cudaq_expectation(observe_results)
    outvariables = [ctx.new_variable()]
    outvals = [result]

    _map(ctx.replace, eqn.outvars, outvariables)
    _map(ctx.write, eqn.outvars, outvals)


def change_namedobs(ctx, eqn):
    """Change named observable to CUDA-quantum equivalent."""
    assert eqn.primitive == namedobs_p

    # Operands to expval_p
    # * qubit
    invals = _map(ctx.read, eqn.invars)
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
    outvariables = [ctx.new_variable()]

    _map(ctx.replace, eqn.outvars, outvariables)
    _map(ctx.write, eqn.outvars, outvals)


def change_hamiltonian(ctx, eqn):
    """Change catalyst hamiltonian to an equivalent expression in CUDA."""
    assert eqn.primitive == hamiltonian_p

    invals = _map(ctx.read, eqn.invars)
    coeffs = invals[0]
    terms = invals[1:]

    hamiltonian = reduce(operator.add, map(operator.mul, coeffs, terms))

    outvariables = [ctx.new_variable()]
    _map(ctx.replace, eqn.outvars, outvariables)
    _map(ctx.write, eqn.outvars, [hamiltonian])


def change_adjoint(ctx, eqn):
    """Change Catalyst adjoint to an equivalent expression in CUDA.

    Do note that this uses concrete evaluation.
    There might be a way to do this with abstract evaluation.
    What do I mean by this?
    Notice that cudaq_make_kernel's API can take types as inputs.
    E.g., (int, float, List, cudaq.qreg)

    These are abstract types or parameters.

    We are currently not passing these inputs to cudaq_make_kernel.

    Instead we are passing only cudaq.qreg and all these inputs are concretely
    known in the adjointed kernel.

    This is not ideal, as for gradients this approach wouldn't work.
    But at the moment we do not support gradients, so we can leave this as future work.
    Let's add this as a TODO.
    TODO(@erick-xanadu): Please add support for abstract types / parameters
    in cudaq.make_kernel.
    """
    assert eqn.primitive == adjoint_p

    # This is the quantum register.
    invals = _map(ctx.read, eqn.invars)
    # By convention this qreg is the last one.
    register = invals[-1]

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
    # invals[0:-1] also includes the consts.
    concrete_args = invals[0:-1] + [abstract_qreg]
    interpreter = InterpreterContext(
        nested_jaxpr.jaxpr,
        nested_jaxpr.literals,
        *concrete_args,
        kernel=kernel_to_adjoint,
    )
    # retval would be the abstract_qreg we passed as an argument here.
    # We always pass the qreg and return the qreg.
    # TODO: Do we support returning any other values?
    _retval = interpret_impl(interpreter, nested_jaxpr.jaxpr)
    cudaq_adjoint(ctx.kernel, kernel_to_adjoint, register)

    # Now that we have captured all of the adjoint...
    # Let's actually create an operation with it.
    _map(ctx.write, eqn.outvars, [register])


def ignore_impl(_ctx, _eqn):
    """No-op"""


def unimplemented_impl(_ctx, eqn):  # pragma: nocover
    """Raise an error."""
    msg = f"{eqn.primitive} is not yet implemented in Catalyst's CUDA-Quantum support."
    raise NotImplementedError(msg)


def default_impl(ctx, eqn):
    """Default implementation for all other non-catalyst primitives. I.e., other JAX primitives."""
    # This little scope was based on eval_jaxpr's implmentation:
    # pylint: disable-next=line-too-long
    #    https://github.com/google/jax/blob/16636f9c97414d0c5195c6fd47227756d4754095/jax/_src/core.py#L507-L518
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    ans = eqn.primitive.bind(*subfuns, *map(ctx.read, eqn.invars), **bind_params)
    if eqn.primitive.multiple_results:  # pragma: nocover
        _map(ctx.write, eqn.outvars, ans)
    else:
        ctx.write(eqn.outvars[0], ans)


INST_IMPL = {
    state_p: change_get_state,
    qextract_p: change_register_getitem,
    qinsert_p: change_register_setitem,
    qinst_p: change_instruction,
    compbasis_p: change_compbasis,
    sample_p: change_sample,
    counts_p: change_counts,
    qmeasure_p: change_measure,
    expval_p: change_expval,
    namedobs_p: change_namedobs,
    hamiltonian_p: change_hamiltonian,
    adjoint_p: change_adjoint,
    # ignore set of instructions we don't care about.
    # because they have been handled before or they are just
    # not necessary in the CUDA-quantum API.
    qdealloc_p: ignore_impl,
    qdevice_p: ignore_impl,
    qalloc_p: ignore_impl,
    # These are unimplemented at the moment.
    zne_p: unimplemented_impl,
    qunitary_p: unimplemented_impl,
    hermitian_p: unimplemented_impl,
    tensorobs_p: unimplemented_impl,
    var_p: unimplemented_impl,
    probs_p: unimplemented_impl,
    cond_p: unimplemented_impl,
    while_p: unimplemented_impl,
    for_p: unimplemented_impl,
    grad_p: unimplemented_impl,
    func_p: unimplemented_impl,
    jvp_p: unimplemented_impl,
    vjp_p: unimplemented_impl,
    print_p: unimplemented_impl,
}


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

    # Main interpreter loop.
    # Note the absense of loops here and branches.
    # Normally in an interpreter loop, we would have
    # while(True):
    for eqn in jaxpr.eqns:
        # This is similar to direct-call threading
        # https://www.cs.toronto.edu/~matz/dissertation/matzDissertation-latex2html/node6.html
        INST_IMPL.get(eqn.primitive, default_impl)(ctx, eqn)

    retvals = _map(ctx.read, jaxpr.outvars)
    for retval in retvals:
        if isinstance(retval, Hashable) and retval in ctx.measurements:
            # pylint: disable-next=line-too-long
            m = "You cannot return measurements directly from a tape when compiling for cuda quantum."
            raise NotImplementedError(m)
    return retvals


class QJIT_CUDAQ:
    """Class representing a just-in-time compiled hybrid quantum-classical function.

    .. note::

        ``QJIT_CUDAQ`` objects are created by the :func:`~.qjit` decorator. Please see
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

        def cudaq_backend_info(device, _capabilities) -> BackendInfo:
            """The extract_backend_info should not be run by the cuda compiler as it is
            catalyst-specific. We need to make this API a bit nicer for third-party compilers.
            """
            device_name = (
                device.target_device.short_name
                if isinstance(device, qml.devices.LegacyDeviceFacade)
                else device.name
            )
            interface_name = (
                device.target_device.name
                if isinstance(device, qml.devices.LegacyDeviceFacade)
                else device.name
            )
            return BackendInfo(device_name, interface_name, "", {})

        with Patcher(
            (QJITDevice, "extract_backend_info", cudaq_backend_info),
            (qml.QNode, "__call__", QFunc.__call__),
        ):
            func = self.user_function
            abs_axes = {}
            static_args = None
            # We could also pass abstract arguments here in *args
            # the same way we do so in Catalyst.
            # But I think that is redundant now given make_jaxpr2
            jaxpr, _, out_treedef, plugins = trace_to_jaxpr(func, static_args, abs_axes, args, {})
            assert not plugins, "Plugins are not compatible with CUDA integration"

        # TODO(@erick-xanadu):
        # What about static_args?
        return jaxpr, out_treedef


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
        if _kwargs:
            # TODO(@erick-xanadu):
            raise NotImplementedError("CUDA tapes do not yet have kwargs.")  # pragma: no cover
        # QJIT_CUDAQ(fun).get_jaxpr
        # will return the JAXPR of the function fun.
        # However, notice that *args are still concrete.
        # This means that when we interpret the JAXPR, it will be a concrete interpretation.
        # If we wanted an abstract interpretation, we could pass the abstract values that
        # correspond to the return value of trace_to_jaxpr out_type2.
        # If we did that, we could cache the result JAXPR from this function
        # and evaluate it each time the function is called.
        catalyst_jaxpr_with_host, out_tree = QJIT_CUDAQ(fun).get_jaxpr(*args)

        # We need to keep track of the consts...
        consts = catalyst_jaxpr_with_host.consts
        catalyst_jaxpr = remove_host_context(catalyst_jaxpr_with_host)
        # TODO(@erick-xanadu): There was a discussion about removing as much as possible the
        # reliance on unstable APIs.
        # Here is jax._src.core.ClosedJaxpr which is another function in the unstable API.
        # Its only use here is to create a new ClosedJaxpr from the variable catalyst_jaxpr typed
        # Jaxpr.
        # Please note that catalyst_jaxpr_with_host is a ClosedJaxpr.
        # But to get it without the host context we need to "open" it up again.
        # And then close it again.
        # pylint: disable-next=line-too-long
        # From the [documentation (paraphrased)](https://jax.readthedocs.io/en/latest/jaxpr.html#understanding-jaxprs)
        #
        #    a closed jaxpr is has two fields the jaxpr and a list of constants.
        #
        # So, a good solution to get rid of this call here is to just interpret the host context.
        # This is not too difficult to do. The only changes would be that we now need to provide
        # semantics for quantum_kernel_p.
        closed_jaxpr = jax._src.core.ClosedJaxpr(catalyst_jaxpr, catalyst_jaxpr.constvars)

        # Because they become args...
        args = list(args) + consts
        args = tuple(args)
        ctx = InterpreterContext(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        out = interpret_impl(ctx, closed_jaxpr.jaxpr)

        out = tree_unflatten(out_tree, out)
        return out

    return wrapped
