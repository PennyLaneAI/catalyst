// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PAULI_FRAME_OPS
#define PAULI_FRAME_OPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

include "Quantum/IR/QuantumDialect.td"

include "PauliFrame/IR/PauliFrameDialect.td"

//===----------------------------------------------------------------------===//
// PauliFrame dialect enums.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// PauliFrame dialect traits.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// PauliFrame dialect attributes.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// PauliFrame dialect operations.
//===----------------------------------------------------------------------===//

def SetRecordOp : PauliFrame_Op<"set_record"> {
    let summary = "Set the Pauli record of the target qubit(s)";
    let description = [{
        Set the Pauli record of the target qubit(s).
    }];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits,
        I1:$x_parity,
        I1:$z_parity
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
        `[` $x_parity `,` $z_parity `]` $in_qubits attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

def ReadRecordOp : PauliFrame_Op<"read_record"> {
    let summary = "Read the Pauli record of the target qubit";
    let description = [{
        Read the Pauli record of the target qubit.
    }];

    let arguments = (ins
        QubitType:$in_qubit
    );

    let results = (outs
        I1:$x_parity,
        I1:$z_parity,
        QubitType:$out_qubit
    );

    let assemblyFormat = [{
        $in_qubit attr-dict `:` type(results)
    }];
}

def CorrectMeasurementOp : PauliFrame_Op<"correct_measurement"> {
    let summary = "Correct the measurement result given the Pauli record";
    let description = [{
        Correct the measurement result given the Pauli record.
    }];

    let arguments = (ins
        I1:$x_parity,
        I1:$z_parity,
        I1:$in_mres
    );

    let results = (outs
        I1:$out_mres
    );

    let assemblyFormat = [{
        `[` $x_parity `,` $z_parity `]` $in_mres attr-dict `:` type(results)
    }];
}

def FlushRecordOp : PauliFrame_Op<"flush_record"> {
    let summary = "Flush the Pauli record(s) of the target qubit(s)";
    let description = [{
        Flush the Pauli record(s) of the target qubit(s).
    }];

    let arguments = (ins
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let assemblyFormat = [{
        $in_qubits attr-dict `:` type(results)
    }];

    let hasVerifier = 1;
}

#endif // PAULI_FRAME_OPS
