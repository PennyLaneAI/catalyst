Resource Effects of Passes
##########################

The ``estimate-transform-sequence`` pass estimates the resources of a program after each pass of
its transform sequence, without running the passes. It analyzes the program once with the resource
analysis, then updates the resources with the *resource effect* of each pass, in order. A pass
without a resource effect stops the estimate with an error, or, with the ``allow-unknown-passes``
option, is assumed not to change the resources.

A resource effect is declared in C++, next to the pass, in ``Catalyst/Analysis/ResourceEffect.h``.

What an effect updates
======================

An effect updates the ``ResourceResult`` of one function at a time, as the resource analysis
computes it:

- ``operations`` counts each operation by its name, number of wires and number of parameters. The
  names are those of ``getResourceName()``, wrapped as ``Adjoint(...)`` and ``<n>C(...)`` for
  adjoint and controlled operations, such as ``PPR-pi/8`` or ``Adjoint(S)``.
- ``numAllocQubits`` and ``peakQubits`` hold the qubits that the function allocates, and the most
  of them live at once.

Loop bodies and callees are functions of their own, weighted by their number of calls when the
functions are combined, so an effect never multiplies by a trip count. Counts may be expected
values, from hints or from the effects of earlier passes. Classical instructions are not updated.

Kinds of effects
================

.. list-table::
    :header-rows: 1

    * - Kind
      - For a pass that
      - Examples
    * - ``makeIdentityEffect()``
      - changes nothing the resource analysis counts
      - ``symbol-dce``
    * - ``makeSubstitutionEffect(rule)``
      - replaces each operation independently of the others
      - ``to-ppr``, ``ppr-to-ppm``, ``gridsynth``
    * - ``makeTransferEffect(transfer)``
      - needs the resources of a whole function, but not their order
      - ``combine-global-phases``

A pass whose result depends on the order of its operations, such as ``cancel-inverses``, cannot
declare an effect yet.

Substitution
------------

A substitution rule is called once per distinct operation of each function. It returns
``std::nullopt`` to keep the operation, or the operations that replace one of it. ``wires.alloc``
qubits are allocated for each replaced operation, and the peak rises by ``wires.peak`` once, since
each replacement releases its qubits before the next one. The effect of ``ppr-to-ppm``, in
``mlir/lib/PBC/Transforms/ppr_to_ppm.cpp``:

.. code-block:: cpp

    static std::optional<Substitution> getPPRToPPMSubstitution(const OpKey &op, const Pass &) {
        if (op.name != "PPR-pi/8" && op.name != "PPR-pi/4") {
            return std::nullopt;
        }
        Substitution substitution{
            {{{"PPM", op.numWires + 1, 0}, 1}, {{"PPM", 1, 0}, 1}, {{"PPR-pi/2", op.numWires, 0}, 1}},
            {/*alloc=*/1, /*peak=*/1}};
        if (op.name == "PPR-pi/8") {
            substitution.ops.push_back({{"pbc.fabricate", 0, 0}, 1});
        }
        return substitution;
    }

Transfer
--------

A transfer function updates the resources of a function in place. The effect of
``combine-global-phases``, in ``mlir/lib/Quantum/Transforms/merge_global_phase.cpp``, keeps one
uncontrolled global phase per function:

.. code-block:: cpp

    static void combineGlobalPhaseResources(ResourceResult &function, const Pass &) {
        double numPhases = 0;
        for (StringRef name : {"GlobalPhase", "Adjoint(GlobalPhase)"}) {
            ...
        }
        if (numPhases > 0) {
            function.operations["GlobalPhase"][{0, 1}] = std::min(numPhases, 1.0);
        }
    }

Pass options
============

An effect receives the pass created from the pass registry, configured with the options of its
``transform.apply_registered_pass`` op. The pass options are protected members of the generated
base class, so the pass exposes what its effect needs through a member function. The second
argument of ``makeSubstitutionEffect`` and ``makeTransferEffect`` verifies the options once per
pass, and emits an error for options the effect does not support:

.. code-block:: cpp

    struct PPRToPPMPass : public impl::PPRToPPMPassBase<PPRToPPMPass> {
        ...
        LogicalResult verifyResourceEffect(Location loc) const {
            if (decomposeMethod != DecomposeMethod::PauliCorrected || avoidYMeasure) {
                return emitError(loc, "...");
            }
            return success();
        }
    };

    ... makeSubstitutionEffect(getPPRToPPMSubstitution, [](const Pass &pass, Location loc) {
        return static_cast<const PPRToPPMPass &>(pass).verifyResourceEffect(loc);
    });

An effect should read the same tables as its pass, so that the two stay in sync. The effect of
``gridsynth`` uses the case tables and loop hints of the pass's patterns.

Registration
============

Each effect has a registration function next to its pass, guarded to run once and declared in the
``Passes.h`` header of its dialect:

.. code-block:: cpp

    void registerPPRToPPMResourceEffect() {
        [[maybe_unused]] static const bool once = [] {
            ResourceEffectRegistry::get().add("ppr-to-ppm", makeSubstitutionEffect(...));
            return true;
        }();
    }

The registry is keyed by the pass argument. Add the function to ``registerAllResourceEffects()``
in ``mlir/include/RegisterAllPasses.h``, which ``catalyst`` and ``quantum-opt`` call.

Testing
=======

Test an effect against its pass: run the pass followed by ``resource-analysis``, and
``estimate-transform-sequence``, on the same input with the same checks. The input holds the
transform sequence with the pass, which the first run does not read. The checks of the second run
start at the level of the pass:

.. code-block:: mlir

    // RUN: quantum-opt --ppr-to-ppm --resource-analysis=output-json %s -o /dev/null | FileCheck %s
    // RUN: quantum-opt --estimate-transform-sequence %s -o /dev/null | FileCheck %s --check-prefixes=ESTIMATE,CHECK

    // ESTIMATE:    "pass": "ppr-to-ppm"
    // CHECK:       "circuit": {
    // CHECK:         "quantum_operations": {
    // ...

See ``mlir/test/PBC/PPRToPPMResourceEffect.mlir`` and the other ``*ResourceEffect.mlir`` tests.
