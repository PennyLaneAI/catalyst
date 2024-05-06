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
compatible programs are supported by default by the Catalyst frontend.

For more information on how to use the frontend, please refer to the
`quickstart guide <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/quick_start.html>`_.

.. frontend-end-inclusion-marker-do-not-remove
