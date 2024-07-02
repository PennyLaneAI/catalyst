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
#pragma once

#include "Gradient/Transforms/Patterns.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace catalyst {
namespace gradient {
void wrapMemRefArgs(func::FuncOp, const TypeConverter *, RewriterBase &, Location, bool = false);
void wrapMemRefArgsFunc(func::FuncOp, const TypeConverter *, RewriterBase &, Location,
                        bool = false);
void wrapMemRefArgsCallsites(func::FuncOp, const TypeConverter *, RewriterBase &, Location,
                             bool = false);
LLVM::GlobalOp insertEnzymeCustomGradient(OpBuilder &builder, ModuleOp moduleOp, Location loc,
                                          func::FuncOp originalFunc, func::FuncOp augmentedPrimal,
                                          func::FuncOp gradient);
} // namespace gradient
} // namespace catalyst
