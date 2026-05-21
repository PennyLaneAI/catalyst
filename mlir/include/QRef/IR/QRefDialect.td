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

#ifndef QREF_DIALECT
#define QREF_DIALECT

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// QRef dialect definition.
//===----------------------------------------------------------------------===//

def QRefDialect : Dialect {
    let summary = "Reference semantics quantum dialect.";
    let description = [{
        A supplemental dialect to the core quantum dialect.

        Quantum operations in this dialect follow reference semantics (as opposed to qubit value
        semantics in the core quantum dialect): the targets of quantum operations in this dialect
        are all qubit references.

        Unlike qubit (or qreg) SSA values in the `Quantum` dialect, a qubit (or qreg) reference SSA
        value in the `QRef` dialect is allowed to be used multiple times. The operands of gates and
        observables will be these qubit (or qreg) reference values.

        For example, in the following circuit, gates and observable ops take in the qubit reference
        they're acting on, and do not produce new qubit values.

        ```mlir
        func.func @expval_circuit() -> f64 {
            %a = qref.alloc(2) : !qref.reg<2>
            %q0 = qref.get %a[0] : !qref.reg<2> -> !qref.bit
            %q1 = qref.get %a[1] : !qref.reg<2> -> !qref.bit
            qref.custom "Hadamard"() %q0 : !qref.bit
            qref.custom "CNOT"() %q0, %q1 : !qref.bit, !qref.bit
            qref.custom "Hadamard"() %q0 : !qref.bit
            %obs = qref.namedobs %q1 [ PauliX] : !quantum.obs
            %expval = quantum.expval %obs : f64
            qref.dealloc %a : !qref.reg<2>
            return %expval : f64
        }
        ```

        Notice that qubit reference values are reusable.
    }];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "qref";

    /// This is the C++ namespace in which the dialect and all of its sub-components are placed.
    let cppNamespace = "::catalyst::qref";

    let dependentDialects = [
        "quantum::QuantumDialect"
    ];

    let useDefaultTypePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// QRef dialect base operation.
//===----------------------------------------------------------------------===//

class QRef_Op<string mnemonic, list<Trait> traits = []> :
        Op<QRefDialect, mnemonic, traits>;


#endif // QREF_DIALECT
