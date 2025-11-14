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

def InitOp : PauliFrame_Op<"init"> {
    let summary = "Initialize the Pauli record of the target qubit(s) to I";
    let description = [{
        Initialize the Pauli record of the target qubit(s) to I. If multiple target qubits are given
        as input, the Pauli record is initialized to I for all the given qubits.
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

def InitQregOp : PauliFrame_Op<"init_qreg"> {
    let summary = "Initialize the Pauli records of the target qubits in the quantum register to I";
    let description = [{
        Initialize the Pauli records of the target qubits in the quantum register to I.
    }];

    let arguments = (ins
        QuregType:$in_qreg
    );

    let results = (outs
        QuregType:$out_qreg
    );

    let assemblyFormat = [{
        $in_qreg attr-dict `:` type(results)
    }];
}

def UpdateOp : PauliFrame_Op<"update"> {
    let summary = "Update the Pauli record of the target qubit(s)";
    let description = [{
        Update the Pauli record of the target qubit(s) given the new Pauli record. The Pauli record
        is encoded by its X- and Z-parity bits. If multiple target qubits are given as input, the
        new Pauli record is used to update the Pauli records of all the given qubits.
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

def ReadOp : PauliFrame_Op<"read"> {
    let summary = "Read the Pauli record of the target qubit";
    let description = [{
        Read the Pauli record of the target qubit. The returned Pauli record is encoded as X- and
        Z-parity bits.
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
        Correct the measurement result given the Pauli record as encoded by its X- and Z-parity bits.
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

def FlushOp : PauliFrame_Op<"flush"> {
    let summary = "Flush the Pauli record(s) of the target qubit(s)";
    let description = [{
        Flush the Pauli record(s) of the target qubit(s). *Flushing* the Pauli record of a qubit
        refers to physically applying the Pauli gates stored in the record on that qubit and then
        resetting its record to I (X=0, Z=0). If multiple target qubits are given as input, the
        Pauli record is flushed on all the given qubits.
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

def SetOp : PauliFrame_Op<"set"> {
    let summary = "Set the Pauli record of the target qubit(s)";
    let description = [{
        Set the Pauli record of the target qubit(s). The Pauli record is encoded by its X- and
        Z-parity bits. If multiple target qubits are given as input, the same Pauli record is
        applied to all the given qubits.

        Note that the `pauli_frame.set` operations is generally not needed in the Pauli frame
        tracking framework as only the `pauli_frame.init` and `pauli_frame.update` operations should
        be used to modify the Pauli record of the target qubit(s). It is included for completeness.
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

#endif // PAULI_FRAME_OPS
