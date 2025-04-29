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

#include <mlir/Dialect/Arith/IR/Arith.h>

#include "QEC/Utils/Utility.h"

namespace catalyst {
namespace qec {

AllocOp buildAllocQreg(Location loc, int64_t size, PatternRewriter &rewriter)
{
    IntegerType intType = rewriter.getI64Type();
    QuregType qregType = QuregType::get(rewriter.getContext());

    TypedAttr typeAttr = rewriter.getIntegerAttr(intType, size);
    IntegerAttr intAttr{};

    auto intVal = rewriter.create<arith::ConstantOp>(loc, typeAttr);
    return rewriter.create<AllocOp>(loc, qregType, intVal, intAttr);
}

ExtractOp buildExtractOp(Location loc, Value qreg, int64_t position, PatternRewriter &rewriter)
{
    auto idxAttr = rewriter.getI64IntegerAttr(position);
    auto type = rewriter.getType<QubitType>();
    return rewriter.create<ExtractOp>(loc, type, qreg, nullptr, idxAttr);
}

LogicalInitKind getMagicState(QECOpInterface op)
{
    int16_t rotationKind = static_cast<int16_t>(op.getRotationKind());
    if (rotationKind > 0) {
        return LogicalInitKind::magic;
    }
    return LogicalInitKind::magic_conj;
}

} // namespace qec
} // namespace catalyst
