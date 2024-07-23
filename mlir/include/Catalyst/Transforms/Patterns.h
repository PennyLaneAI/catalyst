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

#include "Catalyst/Transforms/TBAAUtils.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace catalyst {

void populateBufferizationPatterns(mlir::TypeConverter &, mlir::RewritePatternSet &);

void populateScatterPatterns(mlir::RewritePatternSet &);

void populateHloCustomCallPatterns(mlir::RewritePatternSet &);

void populateQnodeToAsyncPatterns(mlir::RewritePatternSet &);

void populateDisableAssertionPatterns(mlir::RewritePatternSet &);

void populateGEPInboundsPatterns(mlir::RewritePatternSet &);

void populateTBAATagsPatterns(TBAATree &, mlir::LLVMTypeConverter &, mlir::RewritePatternSet &);

} // namespace catalyst
