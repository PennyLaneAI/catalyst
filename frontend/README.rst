.. frontend-start-inclusion-marker-do-not-remove

Catalyst Frontend for PennyLane
###############################

The Catalyst frontend for PennyLane enables just-in-time (JIT) and ahead-of-time (AOT) compilation
of PennyLane and JAX Python programs. It provides an alternative compilation and device execution
pipeline for PennyLane via the use of a jitting function decorator. The frontend employs a
trace-based approach to capture classical and quantum instructions into a computational graph that
is stored in the internal JAXPR representation. The JAX library and custom MLIR Python bindings are
then used to lower from JAXPR to the quantum MLIR representation consumed by the core Catalyst
compiler.

The frontend introduces several extensions to the JAXPR primitives in order to natively trace and
represent quantum instructions. Additionally, extensions to the default PennyLane package enable
compilation of arbitrary control flow inside of quantum functions, as well as support of real-time
mid-circuit measurements and measurement result feedback. Any ``jax.jit`` compatible programs are
default supported by the Catalyst frontend.

For more information on how to use the frontend, please refer to the
`documentation <https://docs.pennylane.ai/projects/catalyst>`_.

Contents
========

The ``catalyst`` Python package is a mixed Python package which relies on some C extensions in the
form of the ``jaxlib`` package and the MLIR Python bindings. It is structured as followed:

- `python_bindings <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/python_bindings>`_ :
    A copy of the auto-generated Python bindings for operations of various MLIR dialects.
    Slated for removal.

- `utils <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/utils>`_ :
    Contains various utility code for the project.

- `compilation_pipelines.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/compilation_pipelines.py>`_ :
    This module contains classes and decorators for just-in-time and ahead-of-time compilation of
    hybrid quantum-classical functions using Catalyst.

- `compiler.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/compiler.py>`_ :
    This module contains functions for lowering, compiling, and linking MLIR/LLVM representations.

- `jax_primitives.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/jax_primitives.py>`_ :
    This module contains JAX-compatible quantum primitives to support the lowering of quantum
    operations, measurements, and observables to JAXPR.

- `jax_tape.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/jax_tape.py>`_ :
    This module contains the :class:`~.JaxTape`, a PennyLane :class:`~.QuantumTape` that supports
    capturing classical computations and control flow of quantum operations that occur within the
    circuit.

- `jax_tracer.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/jax_tracer.py>`_ :
    This module contains functions for tracing and lowering JAX code to MLIR.

- `param_evaluator.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/param_evaluator.py>`_ :
    This module is responsible for stitching JAXPR pieces together by transferring traced values
    produced in piece of JAXPR to another.

- `pennylane_extensions.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/pennylane_extensions.py>`_ :
    This module contains various functions for enabling Catalyst functionality (such as mid-circuit
    measurements, advanced control flow, and gradients) from PennyLane while using :func:`~.qjit`.

.. frontend-end-inclusion-marker-do-not-remove
