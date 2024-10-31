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

// def NamedObservable : I32EnumAttr<"NamedObservable",
//     "Known named observables",
//     [
//         I32EnumAttrCase<"Identity", 0>,
//         I32EnumAttrCase<"PauliX",   1>,
//         I32EnumAttrCase<"PauliY",   2>,
//         I32EnumAttrCase<"PauliZ",   3>,
//         I32EnumAttrCase<"Hadamard", 4>,
//     ]> {
//     let cppNamespace = "catalyst::ion";
//     let genSpecializedAttr = 0;
// }


//===----------------------------------------------------------------------===//
// Ion dialect traits.
//===----------------------------------------------------------------------===//

// def Unitary : NativeOpTrait<"UnitaryTrait">;
// def Hermitian : NativeOpTrait<"HermitianTrait">;

// def IonMemory : Resource<"IonMemory">;

//===----------------------------------------------------------------------===//
// Ion dialect attributes.
//===----------------------------------------------------------------------===//

def NamedObservableAttr : EnumAttr<IonDialect, NamedObservable, "named_observable">;

//===----------------------------------------------------------------------===//
// Ion dialect operations.
//===----------------------------------------------------------------------===//

class Ion_Op<string mnemonic, list<Trait> traits = []> : Op<IonDialect, mnemonic, traits>;

def InitializeOp : Ion_Op<"init"> {
    let summary = "Initialize the Ion runtime.";

    let assemblyFormat = [{
        attr-dict
    }];
}

def FinalizeOp : Ion_Op<"finalize"> {
    let summary = "Teardown the Ion runtime.";

    let assemblyFormat = [{
        attr-dict
    }];
}

#endif // ION_OPS
