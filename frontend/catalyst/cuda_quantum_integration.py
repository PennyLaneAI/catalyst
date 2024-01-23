# JAX primitives for cuda quantum

import jax
import cudaq
from functools import wraps

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
    return jax.numpy.asarray(cudaq.get_state(kernel))

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


def make_primitive_for_gate(gate: str):
    kernel_gate_p = jax.core.Primitive(f"kernel_{gate}")
    kernel_gate_p.multiple_results = True
    method = getattr(cudaq.Kernel, gate)

    def gate_func(kernel, target):
        kernel_gate_p.bind(kernel, target)
        return tuple()

    @kernel_gate_p.def_impl
    def gate_impl(kernel, target):
        method(kernel, target)
        return tuple()

    @kernel_gate_p.def_abstract_eval
    def gate_abs(kernel, target):
        return tuple()

    cgate = f"c{gate}"
    kernel_cgate_p = jax.core.Primitive(f"kernel_{cgate}")
    kernel_cgate_p.multiple_results = True
    cmethod = getattr(cudaq.Kernel, cgate)

    def cgate_func(kernel, control, target):
        kernel_cgate_p.bind(kernel, control, target)

    @kernel_cgate_p.def_impl
    def cgate_impl(kernel, control, target):
        cmethod(kernel, control, target)
        return tuple()

    @kernel_cgate_p.def_abstract_eval
    def cgate_abs(kernel, control, target):
        return tuple()

    return kernel_gate_p, kernel_cgate_p


x_p, cx_p = make_primitive_for_gate("x")
y_p, cy_p = make_primitive_for_gate("y")
z_p, cz_p = make_primitive_for_gate("z")
h_p, ch_p = make_primitive_for_gate("h")
s_p, cs_p = make_primitive_for_gate("s")
t_p, ct_p = make_primitive_for_gate("t")


def make_primitive_for_pgate(gate: str):
    kernel_gate_p = jax.core.Primitive(f"kernel_{gate}")
    kernel_gate_p.multiple_results = True
    method = getattr(cudaq.Kernel, gate)

    def gate_func(kernel, param, target):
        kernel_gate_p.bind(kernel, param, target)
        return tuple()

    @kernel_gate_p.def_impl
    def gate_impl(kernel, param, target):
        method(kernel, param, target)
        return tuple()

    @kernel_gate_p.def_abstract_eval
    def gate_abs(kernel, param, target):
        return tuple()

    cgate = f"c{gate}"
    kernel_cgate_p = jax.core.Primitive(f"kernel_{cgate}")
    kernel_cgate_p.multiple_results = True
    cmethod = getattr(cudaq.Kernel, cgate)

    def cgate_func(kernel, param, control, target):
        kernel_cgate_p.bind(kernel, param, control, target)

    @kernel_cgate_p.def_impl
    def cgate_impl(kernel, param, control, target):
        cmethod(kernel, param, control, target)
        return tuple()

    @kernel_cgate_p.def_abstract_eval
    def cgate_abs(kernel, param, control, target):
        return tuple()

    return kernel_gate_p, kernel_cgate_p


rx_p, crx_p = make_primitive_for_pgate("rx")
ry_p, cry_p = make_primitive_for_pgate("ry")
rz_p, crz_p = make_primitive_for_pgate("rz")
r1_p, cr1_p = make_primitive_for_pgate("r1")


def make_primitive_for_m(gate: str):
    kernel_gate_p = jax.core.Primitive(f"kernel_{gate}")
    method = getattr(cudaq.Kernel, gate)

    def gate_func(kernel, target):
        return kernel_gate_p.bind(kernel, target)

    @kernel_gate_p.def_impl
    def gate_impl(kernel, target):
        return method(kernel, target)

    @kernel_gate_p.def_abstract_eval
    def gate_abs(kernel, target):
        return AbsCudaValue()

    return kernel_gate_p


mx_p = make_primitive_for_m("x")
my_p = make_primitive_for_m("y")
mz_p = make_primitive_for_m("z")


cudaq_sample_p = jax.core.Primitive("cudaq_sample")


def cudaq_sample(kernel, *args, shots_count=1000):
    return cudaq_sample_p.bind(kernel, *args, shots_count=shots_count)


@cudaq_sample_p.def_impl
def cudaq_sample_impl(kernel, *args, shots_count=1000):
    return cudaq.sample(kernel, *args, shots_counts=shots_counts)


@cudaq_sample_p.def_abstract_eval
def cudaq_sample_abs(kernel, *args, shots_counts=1000):
    return AbsCudaSampleResult()


# SKIP Async for the time being
# SKIP observe (spin_operator map is unclear at the moment)
# SKIP VQE
# Ignore everything else?

# There are no lowerings because we will not generate MLIR

def transform_jaxpr_to_cuda_jaxpr(jaxpr, consts, *args):
    from jax._src.util import safe_map
    from catalyst.jax_primitives import qdevice_p, qalloc_p, state_p, qextract_p, qinst_p, compbasis_p, qinsert_p, qdealloc_p

    ignore = { compbasis_p, qinsert_p, qdealloc_p }

    env = {}
    variable_map = {}
    def read(var):
        if type(var) is jax.core.Literal:
            return var.val
        if variable_map.get(var):
            var = variable_map[var]
        return env[var]

    def write(var, val):
        if variable_map.get(var):
           var = variable_map[var]
        env[var] = val

    def replace(original, new):
        variable_map[original] = new


    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    count = 100

    for idx, eqn in enumerate(jaxpr.eqns):
         if eqn.primitive == qdevice_p:
             # qdevice_p takes no parameters so no need to read
             # but it does return a new variable, which is not normally allocated.
             # So we need to create a new variable.
             #outvals = [cudaq_make_kernel_primitive_impl()]
             outvals = [cudaq_make_kernel()]
             kernel = outvals[0]
             outvars = [jax._src.core.Var(count, "", AbsCudaKernel())]
             safe_map(write, outvars, outvals)
             
         elif eqn.primitive == qalloc_p:
             # qalloc reads
             invals = safe_map(read, eqn.invars)

             # We know that invals is actually just 1
             #outvals = [kernel_qalloc_primitive_impl(kernel, invals[0])]
             outvals = [kernel_qalloc(kernel, invals[0])]
             qreg = outvals[0]
             # And we need a different type
             outvars = [jax._src.core.Var(count, "", AbsCudaQReg())]
             
             # qalloc_p does return something, so we want to substitute it
             safe_map(replace, eqn.outvars, outvars)
             safe_map(write, eqn.outvars, outvals)

         elif eqn.primitive == state_p:
             # state_p does take an input, but we don't care about the input
             # here. At least not now for this experiment.
             #outvals = [cudaq_get_state_primitive_impl(kernel)]
             outvals = [cudaq_getstate(kernel)]
             outvars = [jax._src.core.Var(count, "", jax.core.ShapedArray((2,), jax.numpy.complex128))]
             out_state = outvals[0]
             safe_map(replace, eqn.outvars, outvars)
             safe_map(write, eqn.outvars, outvals)

         elif eqn.primitive == qextract_p:
             # extract_p does take an input and we would like to depend on it
             invals = safe_map(read, eqn.invars)
             #outvals = [qreg_getitem_primitive_impl(qreg, invals[1])]
             outvals = [qreg_getitem(qreg, invals[1])]
             outvars = [jax._src.core.Var(count, "", AbsCudaQbit())]

             safe_map(replace, eqn.outvars, outvars)
             safe_map(write, eqn.outvars, outvals)

             # We also need to override the first variable that corresponds to
             # The first invars, which is the qubit

         elif eqn.primitive == qinst_p:
             # Assume qinst_p is just qml.RX for the time being
             invals = safe_map(read, eqn.invars)
             #none = kernel_rx_primitive_impl(kernel, invals[1], invals[0])
             none = kernel_rx(kernel, invals[1], invals[0])
             # Just so that we never use it?
             # safe_map(replace, eqn.outvars, [UnavailableToken])
             # We don't even need to write anything here...

         elif eqn.primitive in ignore:
             continue

         # Do the normal interpretation...
         else:
             subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
             ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
             if eqn.primitive.multiple_results:
                 safe_map(write, eqn.outvars, ans)
             else:
                 write(eqn.outvars[0], ans)
         count += 1


    return safe_map(read, jaxpr.outvars)

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
        return out[0]

    return wrapped
