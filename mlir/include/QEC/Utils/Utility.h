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

#include "QEC/IR/QECDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

namespace catalyst {
namespace qec {

// build quantum.alloc operation.
AllocOp buildAllocQreg(Location loc, int64_t size, PatternRewriter &rewriter);

// build quantum.extract operation.
ExtractOp buildExtractOp(Location loc, Value qreg, int64_t position, PatternRewriter &rewriter);

// get the magic state from the operation.
LogicalInitKind getMagicState(QECOpInterface op);

} // namespace qec
} // namespace catalyst
