// Copyright @@@year@@@ Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "@@@NewDialect@@@/IR/@@@NewDialect@@@Ops.h"
#include "@@@NewDialect@@@/Transforms/Patterns.h"

using namespace mlir;

namespace catalyst {
namespace @@@new_dialect@@@{

#define GEN_PASS_DEF_@@@NEW_DIALECT@@@CONVERSIONPASS
#include "@@@NewDialect@@@/Transforms/Passes.h.inc"

    // [Insert any type converters here]

    // [Insert any operation-conversion passes here]

} // namespace @@@new_dialect@@@
} // namespace catalyst
