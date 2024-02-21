.. frontend-start-inclusion-marker-do-not-remove

Catalyst Frontend for PennyLane
###############################

The Catalyst frontend for PennyLane enables just-in-time (JIT) and ahead-of-time (AOT) compilation
of PennyLane and JAX Python programs. It provides an alternative compilation and device execution
pipeline for PennyLane which is enabled via a custom function decorator. The Catalyst frontend
employs a two-step process for this task:

- The JAX tracing framework is used to capture classical and quantum instructions into a
  computational graph that is stored in JAX's internal program representation (JAXPR).

- The JAX support library and custom MLIR Python bindings are then used to lower from JAXPR to the
  quantum MLIR representation consumed by the Catalyst compiler core.

To facilitate the above process, the frontend introduces several extensions to the JAXPR primitives
in order to natively trace and represent quantum instructions. Additionally, extensions to the
PennyLane package enable compilation of arbitrary control flow inside of quantum functions, as well
as support of real-time mid-circuit measurements and measurement result feedback. Any ``jax.jit``
compatible programs are default supported by the Catalyst frontend.

For more information on how to use the frontend, please refer to the
`quickstart guide <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/quick_start.html>`_.

Contents
========

The ``catalyst`` Python package is a mixed Python package which relies on some C extensions from the
``jaxlib`` package and the MLIR Python bindings. It is structured as follows, with two sub-packages:

- `python_bindings <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/python_bindings>`_:
    A copy of the auto-generated Python bindings for operations of various MLIR dialects.
    Slated for removal.

- `utils <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/utils>`_:
    Contains various utility code for the project.

and the following modules:

- `jit.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/jit.py>`_:
    This module contains classes and decorators for just-in-time and ahead-of-time compilation of
    hybrid quantum-classical functions using Catalyst.

- `compiler.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/compiler.py>`_:
    This module contains functions for lowering, compiling, and linking MLIR/LLVM representations.

- `jax_primitives.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/jax_primitives.py>`_:
    This module contains JAX-compatible quantum primitives to support the lowering of quantum
    operations, measurements, and observables to JAXPR.

- `jax_tape.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/jax_tape.py>`_:
    This module contains a wrapper around the PennyLane :class:`~pennylane.QuantumTape` class that
    supports capturing classical computations and control flow of quantum operations that occur
    within the circuit.

- `jax_tracer.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/jax_tracer.py>`_:
    This module contains functions for tracing and lowering JAX code to MLIR.

- `param_evaluator.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/param_evaluator.py>`_:
    This module is responsible for stitching JAXPR pieces together by transferring traced values
    produced in piece of JAXPR to another.

- `pennylane_extensions.py <https://github.com/PennyLaneAI/catalyst/tree/main/frontend/pennylane_extensions.py>`_:
    This module contains various functions for enabling Catalyst functionality (such as mid-circuit
    measurements, advanced control flow, and gradients) from PennyLane while using :func:`~.qjit`.

.. frontend-end-inclusion-marker-do-not-remove
