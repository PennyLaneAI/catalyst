// Copyright 2024 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ION_OPS
#define ION_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

include "Ion/IR/IonDialect.td"
include "Ion/IR/IonInterfaces.td"
include "Quantum/IR/QuantumDialect.td"

//===----------------------------------------------------------------------===//
// Ion dialect enums.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Ion dialect traits.
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Ion dialect attributes.
//===----------------------------------------------------------------------===//

class Ion_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Ion_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def LevelAttr : Ion_Attr<"Level", "level"> {
  let summary = "A class to represent an atomic level.";

  let parameters = (ins
    "mlir::StringAttr":$label,
    "mlir::IntegerAttr":$principal,
    "mlir::FloatAttr":$spin,
    "mlir::FloatAttr":$orbital,
    "mlir::FloatAttr":$nuclear,
    "mlir::FloatAttr":$spin_orbital,
    "mlir::FloatAttr":$spin_orbital_nuclear,
    "mlir::FloatAttr":$spin_orbital_nuclear_magnetization,
    "mlir::FloatAttr":$energy
  );


  let builders = [
    AttrBuilderWithInferredContext<(ins     
    "mlir::StringAttr":$label,
    "mlir::IntegerAttr":$principal,
    "mlir::FloatAttr":$spin,
    "mlir::FloatAttr":$orbital,
    "mlir::FloatAttr":$nuclear,
    "mlir::FloatAttr":$spin_orbital,
    "mlir::FloatAttr":$spin_orbital_nuclear,
    "mlir::FloatAttr":$spin_orbital_nuclear_magnetization,
    "mlir::FloatAttr":$energy), [{
      return $_get(principal.getContext(), label, principal, spin, orbital, nuclear, spin_orbital, spin_orbital_nuclear, spin_orbital_nuclear_magnetization, energy);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

def TransitionAttr : Ion_Attr<"Transition", "transition"> {
  let summary = "A class to represent a atomic transition between two levels.";

  let parameters = (ins
    "mlir::StringAttr":$level_0,
    "mlir::StringAttr":$level_1,
    "mlir::FloatAttr":$einstein_a,
    "mlir::StringAttr":$multipole
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "mlir::StringAttr":$level_0,
                                        "mlir::StringAttr":$level_1,
                                        "mlir::FloatAttr":$einstein_a,
                                        "mlir::StringAttr":$multipole), [{
      return $_get(einstein_a.getContext(), level_0, level_1, einstein_a, multipole);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

def BeamAttr : Ion_Attr<"Beam", "beam"> {
  let summary = "A class to represent a laser beam.";

  let parameters = (ins
    OptionalParameter<"mlir::IntegerAttr">:$transition_index,
    "mlir::FloatAttr":$rabi,
    "mlir::FloatAttr":$detuning,
    "mlir::DenseI64ArrayAttr": $polarization,
    "mlir::DenseI64ArrayAttr": $wavevector
  );


  let assemblyFormat = "`<` struct(params) `>`";
}

def PhononAttr : Ion_Attr<"Phonon", "phonon"> {
  let summary = "A class to represent a Phonon mode.";

  let parameters = (ins
    "mlir::FloatAttr":$energy,
    "mlir::DenseF64ArrayAttr": $eigenvector
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def PhononArrayAttr : TypedArrayAttrBase<PhononAttr,
                         PhononAttr.summary # " array"> {
  let constBuilderCall = ?;
}

def LevelArrayAttr : TypedArrayAttrBase<LevelAttr,
                         LevelAttr.summary # " array"> {
  let constBuilderCall = ?;
}


def TransitionArrayAttr : TypedArrayAttrBase<TransitionAttr,
                         TransitionAttr.summary # " array"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// Ion dialect operations.
//===----------------------------------------------------------------------===//

def IonOp : Ion_Op<"ion"> {
  let summary = "A class to represent an ion.";

  let arguments = (ins
    Builtin_StringAttr:$name,
    Builtin_FloatAttr:$mass,
    Builtin_FloatAttr:$charge,
    DenseF64ArrayAttr: $position,
    LevelArrayAttr: $levels,
    TransitionArrayAttr: $transitions
  );

  let results = (outs
    IonType:$out_ion
  );

  let assemblyFormat = [{
    attr-dict `:` type($out_ion)
  }];
}

def ModesOp : Ion_Op<"mode"> {
  let summary = "A class to represent an Phonon modes of the system.";

  let arguments = (ins
    PhononArrayAttr: $modes
  );

  let assemblyFormat = [{
    attr-dict
  }];
}


def PulseOp : Ion_Op<"pulse"> {
    let summary = "Represent a pulse (a laser beam and some time).";

    let arguments = (ins
        AnyFloat: $time,
        QubitType: $in_qubit,
        BeamAttr: $beam,
        Builtin_FloatAttr: $phase
    );

    let results = (outs
    PulseType:$out_pulse
    );

    let assemblyFormat = [{
        `(` $time `:` type($time) `)` $in_qubit attr-dict `:` type($out_pulse)
    }];
}


def ParallelProtocolOp : Ion_Op<"parallelprotocol", [SingleBlockImplicitTerminator<"YieldOp">]> {
    let summary = "Represent a parallel protocol of pulses.";

    let arguments = (ins
        Variadic<QubitType>: $in_qubits
    );

    let results = (outs Variadic<QubitType>:$out_qubits);
    let regions = (region SizedRegion<1>:$region);

    let builders = [
      OpBuilder<(ins
        CArg<"mlir::ValueRange", "std::nullopt">:$in_qubits,
        CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>",
            "nullptr">)>
    ];

    let extraClassDeclaration = [{
      using BodyBuilderFn =
          llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>;

    }];

    let assemblyFormat = [{
        `(` $in_qubits `)` attr-dict `:` type($out_qubits) $region
    }];
}

def YieldOp : Ion_Op<"yield", [Pure, ReturnLike, Terminator, ParentOneOf<["ParallelProtocolOp"]>]> {
    let summary = "Return results from parallel protocol regions";

    let arguments = (ins
        Variadic<QubitType>:$results
    );

    let assemblyFormat = [{
        attr-dict ($results^ `:` type($results))?
    }];

    let builders = [
        OpBuilder<(ins), [{ /* nothing to do */ }]>
    ];
}


#endif // ION_OPS
