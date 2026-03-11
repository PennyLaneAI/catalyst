---
name: implement-openqasm-backend
description: Use when implementing the C++ backend to translation Catalyst MLIR to OpenQASM 3.0.
---

# Implement OpenQASM Backend

## Overview
This skill guides the implementation of `TranslateToQASM3.cpp`, part of the Catalyst compiler backend that translates MLIR `quantum`, `scf`, and `arith` dialects into OpenQASM 3.0 source code.

## When to Use
- Translating MLIR Modules to OpenQASM text.
- Implementing the `translateModuleToOpenQASM3` function.
- Registering a new translation with `mlir::TranslationRegistration`.
- Handling specific MLIR operations (`scf.for`, `quantum.custom`).

## Key Requirements
1.  **Translation Framework**: Use `mlir::TranslationRegistration`.
2.  **Functionality**: Implement `mlir::LogicalResult translateModuleToOpenQASM3(mlir::ModuleOp module, llvm::raw_ostream &output)`.
3.  **Visiting Logic**:
    *   Walk the MLIR module using `module.walk(...)` or an `OpVisitor` pattern.
    *   **Module Scope**: Emit OpenQASM header (`OPENQASM 3.0; include "stdgates.inc";`).
    *   **Functions**: Convert `func.func` to `box` (if applicable) or simply emit contents if it's the main entry point.
    *   **Control Flow (SCF)**: Convert `scf::ForOp` to OpenQASM `for` loops.
        *   Extract loop bounds (start, stop, step) from `arith::ConstantOp` operands if possible, or support dynamic bounds.
    *   **Quantum Operations**:
        *   `quantum::CustomOp` (e.g., "h", "cnot") -> `gate q[i];`
        *   Extract qubit indices from SSA values. **Crucial**: You need a mechanism to map MLIR SSA values (qubits) back to OpenQASM variable names (e.g., `q[0]`).
        *   *Simpler Approach*: Assume a global quantum register `q` and map SSA values to indices if possible, or declare `qubit` variables for each allocation.

## Implementation Pattern

```cpp
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class QASM3Emitter {
public:
    QASM3Emitter(raw_ostream &os) : os(os) {}

    LogicalResult emitModule(ModuleOp module) {
        os << "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";
        // Walk operations...
        return success();
    }

private:
    raw_ostream &os;
    // ... helper methods ...
};
}

LogicalResult translateModuleToOpenQASM3(ModuleOp module, raw_ostream &output) {
    QASM3Emitter emitter(output);
    return emitter.emitModule(module);
}

namespace mlir {
void registerToQASM3Translation() {
    TranslateFromMLIRRegistration registration(
        "mlir-to-qasm3",
        "Translate MLIR to OpenQASM 3.0",
        translateModuleToOpenQASM3,
        [](DialectRegistry &registry) {
            // Register required dialects
            registry.insert<quantum::QuantumDialect>();
            registry.insert<scf::SCFDialect>();
            // ...
        });
}
}
```

## Common Pitfalls
- **Register Translation**: Forget to call `registerToQASM3Translation()` in the tool initialization.
- **Dialect Linking**: Ensure `QuantumDialect` and `SCFDialect` are linked and registered in the `DialectRegistry`.
- **SSA to Variables**: Mapping SSA values (which are transient) to named OpenQASM variables (which are persistent names) requires tracking definitions.
