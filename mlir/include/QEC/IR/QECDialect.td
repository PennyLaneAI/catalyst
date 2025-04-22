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

#ifndef QEC_DIALECT
#define QEC_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

include "Quantum/IR/QuantumDialect.td"
include "QEC/IR/QECOpInterfaces.td"

//===----------------------------------------------------------------------===//
// QEC dialect.
//===----------------------------------------------------------------------===//

def QECDialect : Dialect {
    let summary = "A dialect for quantum error correction circuits.";
    let description = [{
        The QEC dialect extends the Quantum dialect with operations and passes that may be useful
        in quantum error correction schemes, such as the Pauli Product Measurement framework.

        See also [A Game of Surface Codes: Large-Scale Quantum Computing with Lattice Surgery
        ](https://doi.org/10.22331/q-2019-03-05-128).
    }];

    let dependentDialects = [
        "catalyst::quantum::QuantumDialect"
    ];

    /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
    let name = "qec";

    /// This is the C++ namespace that the dialect, and all sub-components, get placed in.
    let cppNamespace = "::catalyst::qec";

    /// Use the default attribute printer/parser hooks, otherwise we would explicitly define them.
    let useDefaultAttributePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// QEC dialect enums.
//===----------------------------------------------------------------------===//

def LogicalInitKind : I32EnumAttr<"LogicalInitKind",
    "The initial state of a logical qubit such as |0⟩, |1⟩, |+⟩, |−⟩, |Y⟩, |-Y⟩, or |T⟩.",
        [
        I32EnumAttrCase<"zero",     0>, // |0⟩
        I32EnumAttrCase<"one",      1>, // |1⟩
        I32EnumAttrCase<"plus",     2>, // |+⟩  = (|0⟩ + |1⟩) / sqrt(2)
        I32EnumAttrCase<"minus",    3>, // |-⟩  = (|0⟩ - |1⟩) / sqrt(2)
        I32EnumAttrCase<"plus_i",   4>, // |Y⟩ = (|0⟩ + i|1⟩) / sqrt(2)
        I32EnumAttrCase<"minus_i",  5>, // |-Y⟩ = (|0⟩ - i|1⟩) / sqrt(2)
        I32EnumAttrCase<"magic",    6>, // |T⟩  = |0⟩ + e^{iπ/4}|1⟩
    ]> {
    let cppNamespace = "::catalyst::qec";
    let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// QEC dialect attributes.
//===----------------------------------------------------------------------===//

def PauliWord : TypedArrayAttrBase<StrAttr, "A product of Pauli operators, aka a Pauli word.">;
def LogicalInit : EnumAttr<QECDialect, LogicalInitKind, "enum">;

//===----------------------------------------------------------------------===//
// QEC dialect operations.
//===----------------------------------------------------------------------===//

class QEC_Op<string mnemonic, list<Trait> traits = []> : Op<QECDialect, mnemonic, traits>;

def PrepareStateOp : QEC_Op<"prepare"> {
    let summary = "Initial state of logical qubit.";
    let description = [{
        Prepares logical qubits in a specific initial quantum state, 
        such as |0⟩, |1⟩, |+⟩, |-⟩, |Y⟩, |-Y⟩, or |T⟩.

        This operation is typically used at the start of a quantum circuit to initialize 
        logical qubits according to the error-corrected encoding scheme.

        Example:
        ```mlir
        %0 = qec.prepare zero %q0 : !quantum.bit
        %1 = qec.prepare magic %q0 : !quantum.bit
        ```
        These prepares the logical qubit in the |0⟩ state and the |T⟩ state respectively.
    }];

    let arguments = (ins
        LogicalInit:$init_state,
        Variadic<QubitType>:$in_qubits
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let builders = [
        OpBuilder<(ins
            "::catalyst::qec::LogicalInitKind":$init_state,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PrepareStateOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*init_state=*/ init_state,
            /*in_qubits=*/ in_qubits);
        }]>
    ];

    let assemblyFormat = [{
      $init_state $in_qubits attr-dict `:` type($out_qubits)
    }];
}

def PPRotationOp : QEC_Op<"ppr", [QECOpInterface, AttrSizedOperandSegments]> {
    let summary = "Pauli Product Rotation on qubits.";

    let description = [{
        The PPRotationOp represents a Pauli product rotation operation on a set of qubits.
        It applies a rotation of the form exp(iθP) where:
        - P is a Pauli product (specified by pauli_product)
        - θ is the rotation angle (specified by rotation_kind in fractions of π)
        
        The operation is characterized by:
        1. A Pauli product (e.g., ["X", "I", "Z"]) specifying which Pauli operators to apply
        2. A rotation kind (in fractions of π) specifying the angle
        3. A list of input qubits to apply the rotation to
        
        The operation returns the same number of qubits as input.
        
        Example:
        ```mlir
        %result = qec.ppr ["X", "I", "Z"](4) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
        ```
        This applies exp(iπ/4 * X⊗I⊗Z) to the three qubits.
    }];

    let arguments = (ins
        PauliWord:$pauli_product,  // The Pauli product to apply (e.g., ["X", "I", "Z"])
        I16Attr:$rotation_kind,    // Rotation angle in fractions of π (e.g., 4 for π/2)
        Variadic<QubitType>:$in_qubits,  // The qubits to apply the rotation to
        Optional<I1>:$condition
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let builders = [
        // Convenience builder with no condition
        OpBuilder<
        (ins
            "::mlir::TypeRange":$out_qubits,
            "::mlir::ArrayAttr":$pauli_product,
            "uint16_t":$rotation_kind,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPRotationOp::build($_builder, $_state,
            /*out_qubits=*/ out_qubits,
            /*pauli_product=*/ pauli_product,
            /*rotation_kind=*/ rotation_kind,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ nullptr);
        }]>,
        // Convenience builder with no condition and rotation kind as an attribute
        OpBuilder<
        (ins
            "::mlir::TypeRange":$out_qubits,
            "::mlir::ArrayAttr":$pauli_product,
            "::mlir::IntegerAttr":$rotation_kind,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPRotationOp::build($_builder, $_state,
            /*out_qubits=*/ out_qubits,
            /*pauli_product=*/ pauli_product,
            /*rotation_kind=*/ rotation_kind,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ nullptr);
        }]>,
        // Convenience builder with no type range
        OpBuilder<
        (ins
            "::mlir::ArrayAttr":$pauli_product,
            "::mlir::IntegerAttr":$rotation_kind,
            "::mlir::ValueRange":$in_qubits,
            "::mlir::Value":$condition
        ),[{
            PPRotationOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*pauli_product=*/ pauli_product,
            /*rotation_kind=*/ rotation_kind,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ condition);
        }]>,
        // Convenience builder with no type range with uint16_t
        OpBuilder<
        (ins
            "::mlir::ArrayAttr":$pauli_product,
            "uint16_t":$rotation_kind,
            "::mlir::ValueRange":$in_qubits,
            "::mlir::Value":$condition
        ),[{
            PPRotationOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*pauli_product=*/ pauli_product,
            /*rotation_kind=*/ rotation_kind,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ condition);
        }]
        >,
        // Convenience builder with ArrayRef<StringRef> with condition
        OpBuilder<
        (ins
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product,
            "uint16_t":$rotation_kind,
            "::mlir::ValueRange":$in_qubits,
            "::mlir::Value":$condition
        ),[{
            PPRotationOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*pauli_product=*/ $_builder.getStrArrayAttr(pauli_product),
            /*rotation_kind=*/ rotation_kind,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ condition);
        }]>
    ];

    let assemblyFormat = [{
      $pauli_product `(` $rotation_kind `)` $in_qubits attr-dict (`cond` `(` $condition^ `)`)? `:` type($out_qubits)
    }];

    let hasVerifier = 1;

    code extraBaseClassDeclaration = [{
        bool isNonClifford(){
            int16_t rotationKind = static_cast<int16_t>(getRotationKind());
            return rotationKind == 8 || rotationKind == -8;  // π/8 rotations are non-Clifford
        };
        bool isClifford(){
            int16_t rotationKind = static_cast<int16_t>(getRotationKind());
            return rotationKind == 4 || rotationKind == -4;  // π/4 rotations are Clifford
        };
    }];
    let extraClassDeclaration = extraBaseClassDeclaration;
}

def PPMeasurementOp : QEC_Op<"ppm", [QECOpInterface, AttrSizedOperandSegments]> {
    let summary = "Pauli Product Measurement on qubits.";

    let description = [{
        The PPMeasurementOp represents a Pauli product measurement operation.
        It measures a set of qubits in the basis specified by a Pauli product.
        
        The operation is characterized by:
        1. A Pauli product (e.g., "XIZ") specifying the measurement basis
        2. A list of input qubits to measure
        
        The operation returns:
        1. A measurement result (1-bit classical value)
        2. The post-measurement state of the qubits
        
        Example:
        ```mlir
        %result, %q0, %q1, %q2 = qec.ppm "XIZ" %q0, %q1, %q2 : (qreg, qreg, qreg) -> (i1, qreg, qreg, qreg)
        ```
        This measures the three qubits in the X⊗I⊗Z basis.
    }];

    let arguments = (ins
        PauliWord:$pauli_product,  // The Pauli product specifying the measurement basis
        DefaultValuedAttr<I16Attr, "1">:$rotation_sign, 
        Variadic<QubitType>:$in_qubits,  // The qubits to measure
        Optional<I1>:$condition
    );

    let results = (outs
        I1:$mres,  // The measurement result (0 or 1)
        Variadic<QubitType>:$out_qubits  // The post-measurement state of the qubits
    );

    let builders = [
        // Convenience builder with no condition
        OpBuilder<
        (ins
            "::mlir::Type":$mres,
            "::mlir::TypeRange":$out_qubits,
            "::mlir::ArrayAttr":$pauli_product,
            "::mlir::IntegerAttr":$rotation_sign,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPMeasurementOp::build($_builder, $_state,
            /*mres=*/ mres,
            /*out_qubits=*/ out_qubits,
            /*pauli_product=*/ pauli_product,
            /*rotation_sign=*/ rotation_sign,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ nullptr);
        }]>,
        // Convenience builder with no condition and rotation sign as an attribute
        OpBuilder<
        (ins
            "::mlir::Type":$mres,
            "::mlir::TypeRange":$out_qubits,
            "::mlir::ArrayAttr":$pauli_product,
            "uint16_t":$rotation_sign,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPMeasurementOp::build($_builder, $_state,
            /*mres=*/ mres,
            /*out_qubits=*/ out_qubits,
            /*pauli_product=*/ pauli_product,
            /*rotation_sign=*/ rotation_sign,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ nullptr);
        }]>,
        // Convenience builder with only pauli product and in_qubits, the type of mres is i1
        OpBuilder<
        (ins
            "::mlir::ArrayAttr":$pauli_product,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPMeasurementOp::build($_builder, $_state,
            /*mres=*/ $_builder.getI1Type(),
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*pauli_product=*/ pauli_product, 
            /*rotation_sign=*/ 1,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ nullptr);
        }]>,
        // Convenience builder with ArrayRef<StringRef>
        OpBuilder<
        (ins
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPMeasurementOp::build($_builder, $_state,
            /*pauli_product=*/ $_builder.getStrArrayAttr(pauli_product),
            /*in_qubits=*/ in_qubits);
        }]>
    ];

    let assemblyFormat = [{
      $pauli_product (`(` $rotation_sign^ `)`)? $in_qubits (`cond` `(` $condition^ `)`)? attr-dict `:` type($out_qubits)
    }];

    let hasVerifier = 1;

    code extraBaseClassDeclaration = [{
        uint16_t getRotationKind(){
            return getRotationSign();
        };
        void setRotationKind(uint16_t attrValue){
            setRotationSign(static_cast<int16_t>(attrValue) < 0 ? -1 : 1);
        }
    }];
    let extraClassDeclaration = extraBaseClassDeclaration;
}

def SelectPPMeasurementOp : QEC_Op<"select.ppm"> {
    let summary = "Conditional Pauli product measurement based on a boolean control.";

    let description = [{
    Conditionally applies and Pauli Product Measurement on qubits.

    Based on the boolean `condition` (type `i1`), this op selects between two Pauli product strings:  
    - If `condition` is 1, applies and measures using `pauli_product`.  
    - If `condition` is 0, uses `else_pauli_product`.  

    The operation returns:
    - `mres`: the result of the Pauli product measurement (`i1`)
    - `out_qubits`: post-measurement qubits

    Example:
    ```mlir
    %m3, %3 = qec.select.ppm (%m1, ["X"], ["Z"]) %1#1 : !quantum.bit
    ```
    If %m1 == 1, applies "X" to %1#1; else applies "Z". Then measures.
    }];

    let arguments = (ins
        I1:$condition,  // The condition qubit
        PauliWord:$pauli_product,
        PauliWord:$else_pauli_product,
        Variadic<QubitType>:$in_qubits  // The qubits to measure
    );

    let results = (outs
        I1:$mres,  // The measurement result (0 or 1)
        Variadic<QubitType>:$out_qubits  // The post-measurement state of the qubits
    );

    let builders = [
        OpBuilder<
        (ins
            "::mlir::Value":$condition,
            "::mlir::ArrayAttr":$pauli_product,
            "::mlir::ArrayAttr":$else_pauli_product,
            "::mlir::ValueRange":$in_qubits
        ),[{
            SelectPPMeasurementOp::build($_builder, $_state,
            /*mres_type=*/ $_builder.getI1Type(),
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*condition=*/ condition,
            /*pauli_product=*/ pauli_product,
            /*else_pauli_product=*/ else_pauli_product,
            /*in_qubits=*/ in_qubits);
        }]>,
        OpBuilder<
        (ins
            "::mlir::Value":$condition,
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product,
            "::mlir::ArrayRef<mlir::StringRef>":$else_pauli_product,
            "::mlir::ValueRange":$in_qubits
        ),[{
            SelectPPMeasurementOp::build($_builder, $_state,
            /*mres_type=*/ $_builder.getI1Type(),
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*condition=*/ condition,
            /*pauli_product=*/ $_builder.getStrArrayAttr(pauli_product),
            /*else_pauli_product=*/ $_builder.getStrArrayAttr(else_pauli_product),
            /*in_qubits=*/ in_qubits);
        }]>
        
    ];

    let assemblyFormat = [{
      `(` $condition `,` $pauli_product `,` $else_pauli_product `)` $in_qubits attr-dict `:` type($out_qubits)
    }];
}
#endif // QEC_DIALECT
