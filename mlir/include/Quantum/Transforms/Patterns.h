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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace catalyst {
namespace quantum {

void populateBufferizationLegality(mlir::TypeConverter &, mlir::ConversionTarget &);
void populateBufferizationPatterns(mlir::TypeConverter &, mlir::RewritePatternSet &);
void populateQIRConversionPatterns(mlir::TypeConverter &, mlir::RewritePatternSet &);
void populateQIREEConversionPatterns(mlir::RewritePatternSet &);
void populateAdjointPatterns(mlir::RewritePatternSet &);
void populateSelfInversePatterns(mlir::RewritePatternSet &);
void populateMergeRotationsPatterns(mlir::RewritePatternSet &);

} // namespace quantum
} // namespace catalyst
