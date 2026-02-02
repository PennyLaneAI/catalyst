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

#ifndef QEC_OPS
#define QEC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"  // for ReturnLike
include "mlir/Interfaces/SideEffectInterfaces.td" // for Pure

include "Quantum/IR/QuantumDialect.td"
include "Quantum/IR/QuantumTypes.td"
include "QEC/IR/QECDialect.td"
include "QEC/IR/QECOpInterfaces.td"

//===----------------------------------------------------------------------===//
// QEC dialect operations.
//===----------------------------------------------------------------------===//

class QEC_Op<string mnemonic, list<Trait> traits = []> : Op<QECDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Attribute constraints
//===----------------------------------------------------------------------===//

// Constraint for rotation_kind: must be one of ±1, ±2, ±4, ±8
// The power-of-2 check works because powers of 2 have exactly one bit set.
def RotationKindConstraint : AttrConstraint<
    CPred<"[&]() {"
          "  auto val = ::llvm::cast<::mlir::IntegerAttr>($_self).getInt();"
          "  auto absVal = val < 0 ? -val : val;"
          "  return absVal >= 1 && absVal <= 8 && (absVal & (absVal - 1)) == 0;"
          "}()">,
    "whose value is ±1, ±2, ±4, or ±8">;

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
        It applies a rotation of the form exp(-iθP) where:
        - P is a Pauli product (specified by pauli_product)
        - θ is the rotation angle (specified by rotation_kind in fractions of π)
        
        The operation is characterized by:
        1. A Pauli product (e.g., ["X", "I", "Z"]) specifying which Pauli operators to apply
        2. A rotation kind (in fractions of π) specifying the angle
        3. A list of input qubits to apply the rotation to
        4. A condition (optional) for the operation to be applied only if the condition is true.
        
        The operation returns the same number of qubits as input.
        
        Example:
        ```mlir
        %c0 = arith.constant true : i1
        %Q_0:3 = qec.ppr ["X", "I", "Z"](4) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
        %Q_1:3 = qec.ppr ["X", "I", "Z"](4) %Q_0[0], %Q_0[1], %Q_0[2] cond(%c0) : !quantum.bit, !quantum.bit, !quantum.bit
        ```
        This applies exp(-iπ/4 * X⊗I⊗Z) to the three qubits.
    }];

    let arguments = (ins
        PauliWord:$pauli_product,  // The Pauli product to apply (e.g., ["X", "I", "Z"])
        ConfinedAttr<I16Attr, [RotationKindConstraint]>:$rotation_kind,    // Rotation angle in fractions of π (e.g., 4 for π/2)
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

    let hasCanonicalizeMethod = 1;

    code extraBaseClassDeclaration = [{
        bool hasPiOverTwoRotation(){
            int16_t rotationKind = static_cast<int16_t>(getRotationKind());
            return rotationKind == 2 || rotationKind == -2;
        };
        bool hasPiOverFourRotation(){
            int16_t rotationKind = static_cast<int16_t>(getRotationKind());
            return rotationKind == 4 || rotationKind == -4;
        };
        bool hasPiOverEightRotation(){
            int16_t rotationKind = static_cast<int16_t>(getRotationKind());
            return rotationKind == 8 || rotationKind == -8;
        };

        bool isNonClifford(){
            return hasPiOverEightRotation();  // π/8 rotations are non-Clifford
        };
        bool isClifford(){
            return hasPiOverTwoRotation() || hasPiOverFourRotation();  // π/2 and π/4 rotations are Clifford
        };
    }];
    let extraClassDeclaration = extraBaseClassDeclaration;
}

def PPRotationArbitraryOp : QEC_Op<"ppr.arbitrary", [QECOpInterface, AttrSizedOperandSegments]> {
    let summary = "Pauli Product Rotation with arbitrary angle.";
    let description = [{
        The PPRotationArbitraryOp represents a Pauli product rotation operation with an arbitrary angle.
        It applies a rotation of the form exp(-iθP) to a set of qubits, where:
        - P is a Pauli product (specified by pauli_product)
        - θ is the rotation angle (specified by arbitrary_angle in radians)
        
        The operation is characterized by:
        1. A Pauli product (e.g., ["X", "I", "Z"]) specifying which Pauli operators to apply
        2. A rotation angle (specified by arbitrary_angle in radians)
        3. A list of input qubits to apply the rotation to
        4. A condition (optional) for the operation to be applied only if the condition is true.

        Example:
        ```mlir
        %const = arith.constant 0.14 : f64
        %result = qec.ppr.arbitrary ["X", "Z"](%const) %q0, %q1 : !quantum.bit, !quantum.bit
        ```
        This applies exp(-i0.14 * X⊗Z) to the two qubits.
    }];

    let arguments = (ins
        PauliWord:$pauli_product,
        F64:$arbitrary_angle,
        Variadic<QubitType>:$in_qubits,
        Optional<I1>:$condition
    );

    let results = (outs
        Variadic<QubitType>:$out_qubits
    );

    let builders = [
        OpBuilder<
        (ins
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product,
            "::mlir::Value":$arbitrary_angle,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPRotationArbitraryOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*pauli_product=*/ $_builder.getStrArrayAttr(pauli_product),
            /*arbitrary_angle=*/ arbitrary_angle,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ nullptr);
        }]>,
        // Convenience builder with no condition
        OpBuilder<
        (ins
            "::mlir::TypeRange":$out_qubits,
            "::mlir::ArrayAttr":$pauli_product,
            "::mlir::Value":$arbitrary_angle,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPRotationArbitraryOp::build($_builder, $_state,
            /*out_qubits=*/ out_qubits,
            /*pauli_product=*/ pauli_product,
            /*arbitrary_angle=*/ arbitrary_angle,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ nullptr);
        }]>,
        // Convenience builder with ArrayRef<StringRef> with condition
        OpBuilder<
        (ins
            "::mlir::TypeRange":$out_qubits,
            "::mlir::ArrayRef<mlir::StringRef>":$pauli_product,
            "::mlir::Value":$arbitrary_angle,
            "::mlir::ValueRange":$in_qubits
        ),[{
            PPRotationArbitraryOp::build($_builder, $_state,
            /*out_qubits=*/ out_qubits,
            /*pauli_product=*/ $_builder.getStrArrayAttr(pauli_product),
            /*arbitrary_angle=*/ arbitrary_angle,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ nullptr);
        }]>,
        // Convenience builder with no type range
        OpBuilder<
        (ins
            "::mlir::ArrayAttr":$pauli_product,
            "::mlir::Value":$arbitrary_angle,
            "::mlir::ValueRange":$in_qubits,
            "::mlir::Value":$condition
        ),[{
            PPRotationArbitraryOp::build($_builder, $_state,
            /*out_qubits=*/ mlir::TypeRange(in_qubits),
            /*pauli_product=*/ pauli_product,
            /*arbitrary_angle=*/ arbitrary_angle,
            /*in_qubits=*/ in_qubits,
            /*condition=*/ condition);
        }]>,
    ];

    let assemblyFormat = [{
        $pauli_product `(` $arbitrary_angle `)` $in_qubits attr-dict (`cond` `(` $condition^ `)`)? `:` type($out_qubits)
    }];

    let hasVerifier = 1;
    let hasCanonicalizeMethod = 1;
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
        %result, %q0:3 = qec.ppm ["X", "I", "Z"] %q0, %q1, %q2 : i1, !quantum.bit, !quantum.bit, !quantum.bit
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
      $pauli_product (`(` $rotation_sign^ `)`)? $in_qubits (`cond` `(` $condition^ `)`)? attr-dict `:` type(results)
    }];

    let hasVerifier = 1;

    code extraBaseClassDeclaration = [{
        uint16_t getRotationKind(){
            return getRotationSign();
        };
        void setRotationKind(uint16_t attrValue){
            setRotationSign(static_cast<int16_t>(attrValue) < 0 ? -1 : 1);
        };
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
    %m3, %3 = qec.select.ppm (%m1, ["X"], ["Z"]) %1#1 : i1, !quantum.bit
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
      `(` $select_switch `,` $pauli_product_0 `,` $pauli_product_1 `)` $in_qubits attr-dict `:` type(results)
    }];
    
    let hasVerifier = 1;
}

def LayerOp : QEC_Op<"layer", [SingleBlockImplicitTerminator<"YieldOp">]> {
    let summary = "A layer operation";

    let description = [{
    The `qec.layer` operation represents a group of PPR/PPM operations that are
    either mutually commutative within the group or act on different qubits.
    
    `qec.layer` operates on carried variables and returns the final values after termination.

    The body region must contain exactly one block that terminates with `qec.yield`.

    Example:
    ```mlir
    func.func @layer(%arg0 : !quantum.bit, %arg1 : i1) -> i1 {
        %m, %0 = qec.layer(%q0 = %arg0, %c = %arg1) : !quantum.bit, i1 {
            %res, %q_1 = qec.ppm ["Z"](4) %q0 cond(%c): !quantum.bit
            qec.yield %res, %q_1 : i1, !quantum.bit
        }
        func.return %m : i1
    }
    ```
    }];

    let arguments = (ins Variadic<AnyType>: $initArgs);

    let results = (outs Variadic<AnyType>:$results);
    let regions = (region SizedRegion<1>:$region);

    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<(ins
        CArg<"mlir::ValueRange", "std::nullopt">:$initArgs,
        CArg<"mlir::ValueRange", "std::nullopt">:$outArgs,
        CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange, mlir::ValueRange)>",
            "nullptr">)>
    ];

    let extraClassDeclaration = [{
      using BodyBuilderFn =
          llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange, mlir::ValueRange)>;

    }];

    let hasCustomAssemblyFormat = 1;
}

def YieldOp : QEC_Op<"yield", [ Pure, ReturnLike, Terminator, ParentOneOf<["LayerOp"]> ]> {
    let summary = "Return results from a layer region";

    let arguments = (ins
        Variadic<AnyType>:$results
    );

    let assemblyFormat = [{
        attr-dict ($results^ `:` type($results))?
    }];

    let builders = [
        OpBuilder<(ins), [{ /* nothing to do */ }]>
    ];
}

#endif // QEC_OPS

