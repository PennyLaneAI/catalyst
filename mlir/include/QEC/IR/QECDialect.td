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
    "The initial state of a logical qubit such as |0⟩, |1⟩, |+⟩, |−⟩, |Y⟩, |-Y⟩, |m⟩, or |m̅⟩.",
    [
        I32EnumAttrCase<"zero",     0>,     // |0⟩ Non-magic state
        I32EnumAttrCase<"one",      1>,     // |1⟩ Non-magic state
        I32EnumAttrCase<"plus",     2>,     // |+⟩  = (|0⟩ + |1⟩) / sqrt(2)  Non-magic state
        I32EnumAttrCase<"minus",    3>,     // |-⟩  = (|0⟩ - |1⟩) / sqrt(2)  Non-magic state
        I32EnumAttrCase<"plus_i",   4>,     // |Y⟩  = (|0⟩ + i|1⟩) / sqrt(2) Non-magic / Magic state
        I32EnumAttrCase<"minus_i",  5>,     // |-Y⟩ = (|0⟩ - i|1⟩) / sqrt(2) Non-magic / Magic state
        I32EnumAttrCase<"magic",    6>,     // |m⟩  = |0⟩ + e^{iπ/4}|1⟩      Magic state
        I32EnumAttrCase<"magic_conj", 7>,   // |m̅⟩  = |0⟩ + e^{-iπ/4}|1⟩     Magic state
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
    let summary = "Initialize existing qubits into a given state.";
    let description = [{
        Prepares non-magic states of logical qubits in a specific initial quantum state, 
        such as |0⟩, |1⟩, |+⟩, |-⟩, |Y⟩, |-Y⟩. 
        The input state of qubits can be any state, not necessarily |0⟩.
        
        By default, when allocating a qubit using `quantum.alloc_qb` or `quantum.alloc`, 
        it is prepared in the |0⟩ state.

        Based on QEC scheme, |Y⟩ (plus_i) and |-Y⟩ (minus_i) can be prepared as transversal 
        operations. Otherwise, if those states are fabricated, `FabricateOp` should be used.

        Magic state such as |m⟩ (magic) and |m̅⟩ (magic_conj) cannot be prepared 
        by this operation, use `FabricateOp` instead.

        Example:
        ```mlir
        %0 = qec.prepare zero %q0 : !quantum.bit
        %1 = qec.prepare one %q1 : !quantum.bit
        %2 = qec.prepare plus %q2 : !quantum.bit
        %3 = qec.prepare minus %q3 : !quantum.bit
        %4 = qec.prepare plus_i %q4 : !quantum.bit
        ```
        These prepares the logical qubit in the |0⟩ state and the |m⟩ state respectively.
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

    let hasVerifier = 1;
}

def FabricateOp : QEC_Op<"fabricate"> {
    let summary = "Fabricate axillary qubits from qubit factories.";
    let description = [{
        The FabricateOp represents a operation that produces/fetches auxiliary qubits from a qubit factory.

        FabricateOp is used to prepare states not normally available in an error correction scheme,
        such as magic states |m⟩ (magic) and |m̅⟩ (magic_conj), or |Y⟩ (plus_i), and |-Y⟩ (minus_i)
        in some instances. For states constructible within the scheme, use `PrepareStateOp` instead.
        Contrary to `PrepareStateOp`, this operation does not operate on existing qubits.

        Example:
        ```mlir
        %0 = qec.fabricate magic: !quantum.bit
        %1 = qec.fabricate magic_conj: !quantum.bit
        %2 = qec.fabricate plus_i: !quantum.bit
        %3 = qec.fabricate minus_i: !quantum.bit
        ```
    }];

    let arguments = (ins
        LogicalInit:$init_state
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let builders = [
        // Convenience builder with single qubit
        OpBuilder<
        (ins
            "::catalyst::qec::LogicalInitKind":$init_state
        ),[{
            FabricateOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange({catalyst::quantum::QubitType::get($_builder.getContext())}),
            /*init_state=*/ init_state);
        }]>,
        // Convenience builder with size of qubits
        OpBuilder<
        (ins
            "::catalyst::qec::LogicalInitKind":$init_state,
            "int":$size_of_qubits
        ),[{
            std::vector<mlir::Type> types;
            for (int i = 0; i < size_of_qubits; i++) {
                types.push_back(catalyst::quantum::QubitType::get($_builder.getContext()));
            }
            mlir::TypeRange out_qubits(types);
            FabricateOp::build($_builder, $_state,
            /*out_qubits=*/ out_qubits,
            /*init_state=*/ init_state);
        }]>
    ];

    let assemblyFormat = [{
        $init_state attr-dict `:` type($out_qubits)
    }];

    let hasVerifier = 1;
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
        1. A Pauli product (e.g., ["X", "I", "Z"]) specifying the measurement basis
        2. A list of input qubits to measure
        
        The operation returns:
        1. A measurement result (1-bit classical value)
        2. The post-measurement state of the qubits
        
        Example:
        ```mlir
        %result, %q0:3 = qec.ppm ["X", "I", "Z"] %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
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
        }]>,
        // Convenience builder with ArrayRef<StringRef> and rotation sign
        OpBuilder<
        (ins
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product,
            "uint16_t":$rotation_sign,
            "::mlir::ValueRange":$in_qubits,
            "::mlir::Value":$condition
        ),[{
            PPMeasurementOp::build($_builder, $_state,
            /*mres=*/ $_builder.getI1Type(),
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*pauli_product=*/ $_builder.getStrArrayAttr(pauli_product), 
            /*rotation_sign=*/ rotation_sign,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ condition);
        }]>,
        // Convenience builder with ArrayRef<StringRef> and rotation sign
        OpBuilder<
        (ins
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product,
            "::mlir::ValueRange":$in_qubits,
            "::mlir::Value":$condition
        ),[{
            PPMeasurementOp::build($_builder, $_state,
            /*mres=*/ $_builder.getI1Type(),
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*pauli_product=*/ $_builder.getStrArrayAttr(pauli_product), 
            /*rotation_sign=*/ 1,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ condition);
        }]>,
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
    let summary = "Multiplexed Pauli product measurement.";

    let description = [{
    Based on the boolean `select_switch` (type `i1`), this op selects between two Pauli product strings:  
    - If `select_switch` is 1, applies and measures using `pauli_product_0`.  
    - If `select_switch` is 0, uses `pauli_product_1`.  

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
        I1:$select_switch,
        PauliWord:$pauli_product_0,
        PauliWord:$pauli_product_1,
        Variadic<QubitType>:$in_qubits  // The qubits to measure
    );

    let results = (outs
        I1:$mres,  // The measurement result (0 or 1)
        Variadic<QubitType>:$out_qubits  // The post-measurement state of the qubits
    );

    let builders = [
        OpBuilder<
        (ins
            "::mlir::Value":$select_switch,
            "::mlir::ArrayAttr":$pauli_product_0,
            "::mlir::ArrayAttr":$pauli_product_1,
            "::mlir::ValueRange":$in_qubits
        ),[{
            SelectPPMeasurementOp::build($_builder, $_state,
            /*mres_type=*/ $_builder.getI1Type(),
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*select_switch=*/ select_switch,
            /*pauli_product_0=*/ pauli_product_0,
            /*pauli_product_1=*/ pauli_product_1,
            /*in_qubits=*/ in_qubits);
        }]>,
        OpBuilder<
        (ins
            "::mlir::Value":$select_switch,
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product_0,
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product_1,
            "::mlir::ValueRange":$in_qubits
        ),[{
            SelectPPMeasurementOp::build($_builder, $_state,
            /*mres_type=*/ $_builder.getI1Type(),
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*select_switch=*/ select_switch,
            /*pauli_product_0=*/ $_builder.getStrArrayAttr(pauli_product_0),
            /*pauli_product_1=*/ $_builder.getStrArrayAttr(pauli_product_1),
            /*in_qubits=*/ in_qubits);
        }]>
        
    ];

    let assemblyFormat = [{
      `(` $select_switch `,` $pauli_product_0 `,` $pauli_product_1 `)` $in_qubits attr-dict `:` type($out_qubits)
    }];
    
    let hasVerifier = 1;
}
#endif // QEC_DIALECT
