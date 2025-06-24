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

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Ion/Transforms/oqd_database_managers.hpp"

namespace catalyst {
namespace ion {

void populateGatesToPulsesPatterns(mlir::RewritePatternSet &, const OQDDatabaseManager &);
void populateConversionPatterns(mlir::LLVMTypeConverter &typeConverter,
                                mlir::RewritePatternSet &patterns);

} // namespace ion
} // namespace catalyst
