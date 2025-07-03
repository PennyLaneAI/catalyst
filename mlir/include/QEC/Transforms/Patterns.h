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

#pragma once

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Passes.h" // need for DecomposeMethod

namespace catalyst {
namespace qec {

void populateCliffordTToPPRPatterns(mlir::RewritePatternSet &);
void populateCommutePPRPatterns(mlir::RewritePatternSet &, unsigned int maxPauliSize);
void populateMergePPRIntoPPMPatterns(mlir::RewritePatternSet &, unsigned int maxPauliSize);
void populateDecomposeNonCliffordPPRPatterns(mlir::RewritePatternSet &,
                                             DecomposeMethod decomposeMethod, bool avoidYMeasure);
void populateDecomposeCliffordPPRPatterns(mlir::RewritePatternSet &, bool avoidYMeasure);

} // namespace qec
} // namespace catalyst
