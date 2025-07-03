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

#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Quantum dialect declarations.
//===----------------------------------------------------------------------===//

#include "Quantum/IR/QuantumOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Quantum type declarations.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Quantum/IR/QuantumOpsTypes.h.inc"

class QuantumMemory : public mlir::SideEffects::Resource::Base<QuantumMemory> {
    llvm::StringRef getName() final { return "QuantumMemory"; }
};
