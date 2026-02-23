
NOTE: The definitions in this document are subject to change/refinement under the
      [Better containers for quantum programs](https://app.shortcut.com/xanaduai/epic/112058) epic.

### Definitions & Assumptions

The following assumptions are made about the structure of quantum code within Catalyst's program
representation. These assumptions should generally hold at the early stages of compilation where
quantum passes are applied.

- Pennylane QNodes represent a distinct region in a user program within which quantum executions
  take place.
  - A *quantum execution* involves running code on specialized devices (software or hardware)
    capable of quantum computation. Such executions are characterized by the initialization,
    modification, and measurement of quantum states on the device. Only classical information
    can enter or leave the device.
  - A QNode is described by a function with classical inputs and classical outputs. Meausurement
    processes are instructions that convert quantum states to classical information at the end of
    a quantum execution.
  - On many devices, measurement processes require many executions of the *same* program in a tight
    loop, called shots. Such tighly coupled executions can be considered as one 'quantum execution'
    for our purposes.
  - A QNode may implicitly involve multiple quantum executions. While the user expresses a QNode as
    a single function, the requested computation may require multiple distinct quantum executions
    under the hood. To faithfully track the users intention, the program representation must track
    both the scope of a (single) quantum execution as well as the scope of the original QNode
    function written by the user.
- Catalyst represents PennyLane QNodes as separate [module operations](https://mlir.llvm.org/docs/Dialects/Builtin/#builtinmodule-moduleop)
  nested under the root IR module. These nested modules are referred to as *quantum kernels*,
  and clearly separate QNode scopes from the rest of a hybrid workflow (which is purely classical).
  - The quantum kernel maintains the scope originally defined by Python user functions
    decorated with the `pennylane.qnode` decorator.
  - Transformation schedules generated from Python (e.g. when using pass decorators on
    `pennylane.QNode` objecs) are contained within the quantum kernel, under a special anonymous
    module with the `transform.with_named_sequence` attribute. Such transformations are thus
    always scoped to the entire quantum kernel.
  - Besides the transform schedule, quantum kernels contain an arbitrary number of both classical
    [function operations](https://mlir.llvm.org/docs/Dialects/Func/#funcfunc-funcfuncop) and
    functions representing quantum executions (as defined above).
  - The structure of a quantum kernel in the IR is shown below:
    ```mlir
    module @workflow {
      func.func public @jit_workflow() {
        catalyst.launch_kernel @module_circuit::@circuit()
      }

      module @module_circuit {
        module attributes {transform.with_named_sequence} {
          transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
            ...
          }
        }
        func.func public @circuit() attributes {qnode} {
          %c100_i64 = arith.constant 100 : i64
          quantum.device shots(%c100_i64) ["librtd_null_qubit.dylib", "NullQubit"]
          ...
          quantum.device_release
        }
      }
    }
    ```
    We can see the root IR module (`@workflow`), as well as the program entry point function
    (prefixed with `jit_`) which invokes a quantum kernel via the `launch_kernel` operation.
    The quantum kernel is the nested module `@module_circuit`, which carries the transform
    schedule for the QNode scope (anonymous module with the `transform.with_named_sequence`
    attribute), as well as one or more functions. The `launch_kernel` instruction defines which
    function is the starting for the execution of a quantum kernel.
  - NOTE: The quantum kernel abstraction presented here only exists until the `inline-nested-module`
          pass is run. However, most quantum transformations (including anything scheduled from the
          frontend) happen *before* the quantum kernel is lowered to a simpler representation, so
          this should be of little impact to most quantum transform developers.
          (see Catalyst's [default pipeline](https://github.com/PennyLaneAI/catalyst/blob/v0.14.0/frontend/catalyst/pipelines.py#L229))
- Within a quantum kernel, individual quantum executions are represented by functions that
  carry the special `qnode` attribute. We will refer to these as *qnode functions* (name TBD):
  - A precise definition is difficult to provide and may not currently apply to all transformations
    or stages in Catalyst, however, for the purpose of this guide a qnode function will contain:
    - device [initialization](https://github.com/PennyLaneAI/catalyst/blob/v0.14.0/frontend/catalyst/python_interface/dialects/quantum.py#L389)
      and [release](https://github.com/PennyLaneAI/catalyst/blob/v0.14.0/frontend/catalyst/python_interface/dialects/quantum.py#L412)
      instructions
    - one or more [measurement processes](https://github.com/PennyLaneAI/catalyst/blob/v0.14.0/frontend/catalyst/python_interface/dialects/quantum.py#L1058)

    Conversely, such operations should not appear in any other kind of function. Quantum state
    evolution [operations](https://github.com/PennyLaneAI/catalyst/blob/v0.14.0/frontend/catalyst/python_interface/dialects/quantum.py#L273)
    may appear in qnode or non-qnode functions (e.g. as quantum subroutines). The key charactestic
    of a qnode function is that it represents a quantum execution as defined above.
  - By default, a PennyLane QNode will lower to a quantum kernel module with a single qnode
    function within. Subsequent passes however may result in a multiple qnode functions to appear
    within the quantum kernel.
  - Note that a function calling out to another function that contains measurement processes does
    *not* count as a qnode function. Such a function should be considered a classical function.


### Guidelines

During the design of quantum compilation passes targeting Catalyst programs with the above
structure, the following principles should be applied to ensure composability with other passes
and smooth integration into the compilation infrastracture.

- Passes targeting PennyLane QNodes must be written to run on `ModuleOp`s.
  - The reason is that PennyLane QNodes are translated to quantum kernel modules in the IR
    (see assumptions section).
  - This applies to xDSL as well as MLIR passes, since both are scheduled on the quantum kernel.
- A pass is resposible for traversing the module it is scheduled on and identify all qnode functions
  present within (identifiable via the `qnode` attribute).
- Unless otherwise required, each qnode function should be processed once by the pass.
- Unless otherwise required, each qnode function should be processed independently.
- A pass should exit early without error if it cannot find the operations or patterns it is
  designed to target. Execution should only be aborted where IR invariants or Catalyst program
  model expectations are clearly violated.
- Passes that wish to replace an existing qnode function with multiple quantum executions
  or with additional classical processing can do so by:
  - Creating new `FuncOp`s with the `qnode` attribute within the parent module of the targeted
    qnode function. These functions must have a different symbol name than the targeted qnode
    function.
  - The targeted qnode function is modified or replaced with a function calling the newly
    created qnode functions. This function is now a "classical" function and thus must not
    carry the `qnode` attribute anymore. The signature and symbol name of this classical
    function should remain the same, such that existing calls to it remain valid.
  - Arbitrary classical pre- and post-processing may be inserted into the new classical
    function. This is typically important in order to maintain the action of the original
    qnode function throughout transformation. For instance, a pass that turns `expval` MPs
    into `sample` MPs within a qnode function needs to insert classical processing that
    computes the average of the samples produced by the new qnode function.

Concrete examples for the recommended scheme can be found in the following MLIR passes:
- [Split-to-single-terms](https://github.com/PennyLaneAI/catalyst/pull/2441)
  (additional post-processing)
- [Split-non-commuting](https://github.com/PennyLaneAI/catalyst/pull/2437)
  (multiple executions)

Future consideration:
- We may want to consider allowing QNode passes to be scheduled on `FuncOp`s directly, for
  passes that only modify the internals of a QNode (think gate cancellation).
  - This would allow passes to run in parallel on different QNode functions within a
    quantum kernel module.
  - This would require adjustments to the transform interpreter pass, which currently only
    applies each pass once to the quantum kernel module.


### Miscellaneous Notes

This section is not directly relevant to designing quantum passes. However, when writing
MLIR passes there a few different methods such passes can be invoked by the framework, which
are summarized here.

Different ways to apply passes include:
- direct invocation from the commandline via `opt`, e.g. `quantum-opt --my-pass`
- textual pass pipeline invoked from the commandline, e.g.
  `quantum-opt --pass-pipeline='builtin.module(func.func(my-pass))'`
- programmatic invocation via a pass manager, e.g. `PassManager.addPass(MyPass())` and
  `PassManager.addNestedPass<func::FuncOp>(MyPass())`

How passes are invoked on the IR based on the invocation patterns above:
- *Dialect conversion* and *rewrite pattern* passes are typically op-agnostic passes that invoke
  a pattern rewrite driver.
  - The root op for the pass must be isolated from above (e.g. `ModuleOp`, `FuncOp`).
  - The rewrite driver will traverse *all* nesting layers under the root op looking for
    operations that match the given patterns.
- *Op-agnostic* passes will be invoked on whatever they are scheduled on:
  - direct cmd: the (implicit) root module
  - textual pipeline: all ops of the given leaf type at the given nesting layer
- *Filtered op* passes will work on all ops of the declared type at *one* level of nesting
  for instance:
  - direct cmd: MLIR automatically applies the pass to the root module or to the op type
    in the first level under the root module. In particular, it will not descend into nested
    modules.
  - textual pipeline: The given pipeline leaf must match the filter op type. The pass is applied
    to all ops of that type at the specified nesting layer.
