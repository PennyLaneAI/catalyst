// Copyright 2023 Xanadu Quantum Technologies Inc.

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

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"

#include "CompilerDriver.h"

namespace catalyst {
namespace driver {

/// Register the translations needed to convert to LLVM IR.
void registerLLVMTranslations(mlir::DialectRegistry &registry);

mlir::LogicalResult compileObjectFile(const CompilerOptions &options,
                                      std::shared_ptr<llvm::Module> module,
                                      llvm::TargetMachine *targetMachine, llvm::StringRef filename);

} // namespace driver
} // namespace catalyst
