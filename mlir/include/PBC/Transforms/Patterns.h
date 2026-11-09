// Copyright 2025-2026 Xanadu Quantum Technologies Inc.

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

#include "PBC/IR/PBCDialect.h"

#include "PBC/Transforms/PassesEnums.h.inc" // need for DecomposeMethod

namespace catalyst {
namespace pbc {

void populateToPPRPatterns(mlir::RewritePatternSet &);
void populateCommutePPRPatterns(mlir::RewritePatternSet &, unsigned int maxPauliSize);
void populateMergePPRPatterns(mlir::RewritePatternSet &, unsigned int maxPauliSize);
void populateMergePPRIntoPPMPatterns(mlir::RewritePatternSet &, unsigned int maxPauliSize);
void populateDecomposeNonCliffordPPRPatterns(mlir::RewritePatternSet &,
                                             DecomposeMethod decomposeMethod, bool avoidYMeasure,
                                             bool emitGadgets = false);
void populateDecomposeCliffordPPRPatterns(mlir::RewritePatternSet &, bool avoidYMeasure,
                                          bool emitGadgets = false);
void populatePPRToMBQCPatterns(mlir::RewritePatternSet &);
void populateDecomposeArbitraryPPRPatterns(mlir::RewritePatternSet &);
void populateUnrollConditionalPPRPPMPatterns(mlir::RewritePatternSet &);
void populateLowerPBCInitOpsPatterns(mlir::RewritePatternSet &);
/// Flatten every pbc.gadget under `root` into its region's ops, between pbc.gadget_begin and
/// pbc.gadget_end. Run before the LLVM conversion, so that the conversion never maps values across
/// regions.
void flattenGadgets(mlir::Operation *root);
void populateConversionPatterns(mlir::LLVMTypeConverter &typeConverter,
                                mlir::RewritePatternSet &patterns);
} // namespace pbc
} // namespace catalyst
