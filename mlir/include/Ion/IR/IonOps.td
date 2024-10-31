// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef Ion_OPS
#define Ion_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "Ion/IR/IonDialect.td"
include "Ion/IR/IonInterfaces.td"

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
    "mlir::IntegerAttr":$principal,
    "mlir::FloatAttr":$spin,
    "mlir::FloatAttr":$orbital,
    "mlir::FloatAttr":$nuclear,
    "mlir::FloatAttr":$spin_orbital,
    "mlir::FloatAttr":$spin_orbital_nuclear,
    "mlir::FloatAttr":$spin_orbital_nuclear_magnetization,
    "mlir::FloatAttr":$energy), [{
      return $_get(principal.getContext(), principal, spin, orbital, nuclear, spin_orbital, spin_orbital_nuclear, spin_orbital_nuclear_magnetization, energy);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

def PhononAttr : Ion_Attr<"Phonon", "phonon"> {
  let summary = "A class to represent a phonon.";

  let parameters = (ins
    "mlir::FloatAttr":$energy,
    "mlir::IntegerAttr":$eigen_vector,
  );


  let builders = [
    AttrBuilderWithInferredContext<(ins     
    "mlir::FloatAttr":$energy,
    "mlir::DenseIntElementsAttr":$eigen_vector), [{
      return $_get(energy.getContext(), energy, eigen_vector);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

def TransitionAttr : Ion_Attr<"Transition", "transition"> {
  let summary = "A class to represent a atomic transition between two levels.";

  let parameters = (ins
    "LevelAttr":$level_0,
    "LevelAttr":$level_1,
  );


  let builders = [
    AttrBuilderWithInferredContext<(ins     
    "LevelAttr":$level_0,
    "LevelAttr":$level_1,
    "mlir::FloatAttr":$einstein_a,), [{
      return $_get(level_0.getContext(), level_0, level_1, einstein_a);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}


def LevelArrayAttr : TypedArrayAttrBase<
    LevelAttr, "Array of atomic levels.">;

def TransitionArrayAttr : TypedArrayAttrBase<
    TransititionAttr, "Array of transtion between atomic level">;

def IonAttr : Ion_Attr<"Ion", "ion"> {
  let summary = "A class to represent an ion.";

  let parameters = (ins
    "mlir::FloatAttr":$mass,
    "mlir::FloatAttr":$charge,
    "mlir::DenseIntElementsAttr": $position,
    LevelArrayAttr: $levels,
    TransitionArrayAttr: $transitions,
  );


  let builders = [
    AttrBuilderWithInferredContext<(ins     
    "LevelAttr":$level_0,
    "LevelAttr":$level_1,
    "mlir::FloatAttr":$einstein_a,), [{
      return $_get(mass.getContext(), mass, charge, position, levels, transitions);
    }]>
  ];

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// Ion dialect operations.
//===----------------------------------------------------------------------===//

class Ion_Op<string mnemonic, list<Trait> traits = []> : Op<IonDialect, mnemonic, traits>;

def BeamOp : Ion_Op<"beam"> {
    let summary = "Represent a beam.";

    let arguments = (ins
        TransitionAttr:$transition,
        "mlir::FloatAttr":$rabi,
        "mlir::FloatAttr":$detuning,
        "mlir::DenseIntElementsAttr": $polarization,
        "mlir::DenseIntElementsAttr": $wavevector,
        QubitType: $in_qubit,
    );

    let results = (outs
        QubitType:$out_qubit,
    );
    let assemblyFormat = [{
        attr-dict `transition` `rabi` `detuning` `polarization` `wavevector` 
        %in_qubit : type -> %out_qubit : type
    }];
}

def BeamOp : Ion_Op<"beam"> {
    let summary = "Represent a beam.";

    let arguments = (ins
        TransitionAttr:$transition,
        "mlir::FloatAttr":$rabi,
        "mlir::FloatAttr":$detuning,
        "mlir::DenseIntElementsAttr": $polarization,
        "mlir::DenseIntElementsAttr": $wavevector,
        QubitType: $in_qubit,
    );

    let results = (outs
        QubitType:$out_qubit,
    );
    let assemblyFormat = [{
        attr-dict `transition` `rabi` `detuning` `polarization` `wavevector` 
        %in_qubit : type -> %out_qubit : type
    }];
}

// def SequentialProtocolOp : Ion_Op<"sequential_protocol"> {
//     let summary = "Represent beams that are run sequentially.";

// }

// def ParallelProtocolOp : Ion_Op<"parallel_protocol"> {
//     let summary = "Represent beams that are run in parallel";

// }

#endif // ION_OPS
