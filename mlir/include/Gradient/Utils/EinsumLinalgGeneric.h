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

#include "llvm/ADT/SmallVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace catalyst {

/// Declare the two-tensor Einstein summation in terms of the Linalg.generic and Arith dialect
/// operations.
///
/// Parameters `a` and `b` are the tensors to be processed.
///
/// The `axisCodesA`, `axisCodesB` and `axisCodesResult` all correspond to the einsum program
/// found e.g. in the `numpy.einsum` funciton. The items of these arrays encode axis of `a`,
/// `b` and the `result` correspondingly. For example, the `([0,1,2,3], [2,3], [0,1])` codes would
/// be equivalent to the following `np.einsum` format string: `"abcd,cd->ab"`.
mlir::Value einsumLinalgGeneric(mlir::OpBuilder &builder, mlir::Location loc,
                                llvm::ArrayRef<int64_t> axisCodesA,
                                llvm::ArrayRef<int64_t> axisCodesB,
                                llvm::ArrayRef<int64_t> axisCodesResult, mlir::Value a,
                                mlir::Value b, std::optional<mlir::Value> bufferOut = std::nullopt);

} // namespace catalyst
