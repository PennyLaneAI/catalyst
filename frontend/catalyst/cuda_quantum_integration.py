# JAX primitives for cuda quantum

import jax

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

jax.core.pytype_aval_mappings[CudaValue] = lambda x: x.aval
jax.core.pytype_aval_mappings[CudaQReg] = lambda x: x.aval
jax.core.pytype_aval_mappings[CudaQbit] = lambda x: x.aval
core.raise_to_shaped_mappings[AbsCudaValue] = lambda aval, _: aval
core.raise_to_shaped_mappings[AbsCudaQReg] = lambda aval, _: aval
core.raise_to_shaped_mappings[AbsCudaKernel] = lambda aval, _: aval
core.raise_to_shaped_mappings[AbsCudaQbit] = lambda aval, _: aval

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

kernel_qalloc_p     = jax.core.Primitive("kernel_qalloc")

def kernel_qalloc(kernel, size):
    return kernel_qalloc_p.bind(kernel, size)

@kernel_qalloc_p.def_impl
def kernel_qalloc_primitive_impl(kernel, size):
    return kernel.qalloc(size)

@kernel_qalloc_p.def_abstract_eval
def kernel_qalloc_primitive_abs(kernel, size):
    return AbsCudaQReg()

qreg_getitem_p      = jax.core.Primitive("qreg_getitem")

def qreg_getitem(qreg, idx):
    return qreg_getitem_p.bind(qreg, idx)

@qreg_getitem_p.def_impl
def qreg_getitem_primitive_impl(qreg, idx):
   return qreg[idx]

@qreg_getitem_p.def_abstract_eval
def qreg_getitem_primitive_abs(qreg, idx):
   return AbsCudaQbit()

cudaq_get_state_p   = jax.core.Primitive("cudaq_get_state")
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

def make_primitive_for_gate(gate: str)
    kernel_gate_p = jax.core.Primitive(f"kernel_{gate}")
    kernel_gate_p.multiple_results = True
    method = getattr(kernel, gate)

    def gate(kernel, target):
        kernel_gate_p.bind(kernel, target)
        return tuple()

    @kernel_gate_p.def_impl
    def gate_impl(kernel, target):
        method(kernel, target)
        return tuple()

    @kernel_gate_p.abstract_eval
    def gate_abs(kernel, target):
        return tuple()

    cgate = f"c{gate}"
    kernel_cgate_p = jax.core.Primitive(f"kernel_{cgate}")
    kernel_cgate_p.multiple_results = True
    cmethod = getattr(kernel, cgate)
    def cgate(kernel, control, target):
        kernel_cgate_p.bind(kernel, control, target)

    @kernel_cgate_p.def_impl
    def cgate_impl(kernel, control, target):
        cmethod(kernel, control, target)
        return tuple()

    @kernel_cgate_p.abstract_eval
    def cgate_abs(kernel, control, target):
        return tuple()

    return kernel_gate_p, kernel_cgate_p
 
x_p, cx_p = make_primitive_for_gate("x")
y_p, cy_p = make_primitive_for_gate("y")
z_p, cz_p = make_primitive_for_gate("z")
h_p, ch_p = make_primitive_for_gate("h")
s_p, cs_p = make_primitive_for_gate("s")
t_p, ct_p = make_primitive_for_gate("t")

def make_primitive_for_pgate(gate: str)
    kernel_gate_p = jax.core.Primitive(f"kernel_{gate}")
    kernel_gate_p.multiple_results = True
    method = getattr(kernel, gate)

    def gate(kernel, param, target):
        kernel_gate_p.bind(kernel, param, target)
        return tuple()

    @kernel_gate_p.def_impl
    def gate_impl(kernel, param, target):
        method(kernel, param, target)
        return tuple()

    @kernel_gate_p.abstract_eval
    def gate_abs(kernel, param, target):
        return tuple()

    cgate = f"c{gate}"
    kernel_cgate_p = jax.core.Primitive(f"kernel_{cgate}")
    kernel_cgate_p.multiple_results = True
    cmethod = getattr(kernel, cgate)
    def cgate(kernel, param, control, target):
        kernel_cgate_p.bind(kernel, param, control, target)

    @kernel_cgate_p.def_impl
    def cgate_impl(kernel, param, control, target):
        cmethod(kernel, param, control, target)
        return tuple()

    @kernel_cgate_p.abstract_eval
    def cgate_abs(kernel, param, control, target):
        return tuple()

    return kernel_gate_p, kernel_cgate_p

rx_p, crx_p = make_primitive_for_pgate("rx")
ry_p, cry_p = make_primitive_for_pgate("ry")
rz_p, crz_p = make_primitive_for_pgate("rz")
r1_p, cr1_p = make_primitive_for_pgate("r1")

def make_primitive_for_m(gate: str)
    kernel_gate_p = jax.core.Primitive(f"kernel_{gate}")
    method = getattr(kernel, gate)

    def gate(kernel, target):
        return kernel_gate_p.bind(kernel, target)

    @kernel_gate_p.def_impl
    def gate_impl(kernel, target):
        return method(kernel, target)

    @kernel_gate_p.abstract_eval
    def gate_abs(kernel, target):
        return AbsCudaValue()

    return kernel_gate_p

mx_p = make_primitive_for_m("x")
my_p = make_primitive_for_m("y")
mz_p = make_primitive_for_m("z")

