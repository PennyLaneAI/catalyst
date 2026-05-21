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

#pragma once

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"


using namespace mlir;

namespace catalyst::quantum {

class DecompCallback {
  public:
    virtual ~DecompCallback() = default;
    virtual mlir::OwningOpRef<mlir::func::FuncOp>
    lowerPauliRot(mlir::MLIRContext *ctx, double theta,
                  const std::string &pauliWord,
                  llvm::ArrayRef<int> wires) = 0;
};

void registerDecompCallback(std::unique_ptr<DecompCallback>);
DecompCallback *getDecompCallback();

} // namespace catalyst::quantum
