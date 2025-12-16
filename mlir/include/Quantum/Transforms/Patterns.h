// Copyright 2022-2025 Xanadu Quantum Technologies Inc.

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

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/AllocatorBase.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace catalyst {
namespace quantum {

void populateGridsynthPatterns(mlir::RewritePatternSet &patterns, double epsilon, bool pprBasis);
void populateQIRConversionPatterns(mlir::TypeConverter &, mlir::RewritePatternSet &, bool);
void populateAdjointPatterns(mlir::RewritePatternSet &);
void populateCancelInversesPatterns(mlir::RewritePatternSet &);
void populateMergeRotationsPatterns(mlir::RewritePatternSet &);
void populateIonsDecompositionPatterns(mlir::RewritePatternSet &);
void populateDecomposeLoweringPatterns(mlir::RewritePatternSet &,
                                       const llvm::StringMap<mlir::func::FuncOp> &,
                                       const llvm::StringSet<llvm::MallocAllocator> &);
void populateLoopBoundaryPatterns(mlir::RewritePatternSet &, unsigned int mode);

} // namespace quantum
} // namespace catalyst
