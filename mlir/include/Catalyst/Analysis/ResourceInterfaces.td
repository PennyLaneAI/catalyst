// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CATALYST_ANALYSIS_RESOURCE_INTERFACES
#define CATALYST_ANALYSIS_RESOURCE_INTERFACES

include "mlir/IR/OpBase.td"

// Dialect-agnostic interfaces used by ResourceAnalysis. Dialects opt in by
// implementing these interfaces so the analysis does not hard-code op names.
// Each interface maps to one ResourceResult field:
//   ResourceQuantumOpInterface      -> operations[name][{numQubits, numParams}]
//   ResourceAllocQubitOpInterface   -> num_alloc_qubits
//   ResourceMeasurementOpInterface  -> measurements[name]

def ResourceQuantumOpInterface : OpInterface<"ResourceQuantumOpInterface"> {
  let description = [{
    This interface marks operations that ResourceAnalysis should count as
    quantum instructions. Implementing ops report a display name, the number
    of qubits they act on, and the number of classical parameters. The analysis
    accumulates them into `ResourceResult::operations` keyed by
    `(name, numQubits, numParams)`. Region-level adjoint wrapping is handled
    by the analysis; ops only expose their own adjoint flag when present.
  }];

  let cppNamespace = "::catalyst";
  let methods = [
    InterfaceMethod<
        "Return the resource display name for this op (without Adjoint(...) wrapping).",
        "llvm::StringRef", "getResourceName", (ins), /*methodBody*/[{}]
    >,
    InterfaceMethod<
        "Return the number of qubits this op acts on (including controls).",
        "uint64_t", "getResourceNumQubits", (ins), /*methodBody*/[{}]
    >,
    InterfaceMethod<
        "Return the number of classical gate parameters for this op.",
        "uint64_t", "getResourceNumParams", (ins), /*methodBody*/[{}]
    >,
    InterfaceMethod<
        "Return whether this op's own adjoint attribute is set (e.g. unit attr `adj`).",
        "bool", "getResourceAdjointFlag", (ins), /*methodBody*/[{}],
        /*default=*/[{ return false; }]
    >,
  ];
}

def ResourceAllocQubitOpInterface : OpInterface<"ResourceAllocQubitOpInterface"> {
  let description = [{
    This interface marks operations that allocate qubits. ResourceAnalysis
    uses `getResourceNumAllocQubits` to accumulate into `ResourceResult::numAllocQubits`.
  }];

  let cppNamespace = "::catalyst";
  let methods = [
    InterfaceMethod<
        "Return how many qubits this op allocates.",
        "uint64_t", "getResourceNumAllocQubits", (ins), /*methodBody*/[{}]
    >,
  ];
}

def ResourceMeasurementOpInterface : OpInterface<"ResourceMeasurementOpInterface"> {
  let description = [{
    This interface marks measurement operations and processes. ResourceAnalysis
    counts them under `ResourceResult::measurements` using the name returned by
    `getResourceMeasurementName` (for example `MidCircuitMeasure` or
    `expval(PauliZ)`).
  }];

  let cppNamespace = "::catalyst";
  let methods = [
    InterfaceMethod<
        "Return the resource display name for this measurement (including observable info when applicable).",
        "llvm::StringRef", "getResourceMeasurementName", (ins), /*methodBody*/[{}]
    >,
  ];
}

#endif // CATALYST_ANALYSIS_RESOURCE_INTERFACES
