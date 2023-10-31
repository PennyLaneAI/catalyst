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

#include "Zne.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

#include "Quantum/IR/QuantumOps.h"
#include "Mitigation/IR/MitigationOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace catalyst {
namespace mitigation {

LogicalResult ZneLowering::match(mitigation::ZneOp op) const { return success(); }

void ZneLowering::rewrite(mitigation::ZneOp op, PatternRewriter &rewriter) const
{
    Location loc = op.getLoc();
    func::FuncOp fnOp = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
    auto call = rewriter.create<func::CallOp>(loc, fnOp, op.getArgs());
    rewriter.replaceOp(op, call.getResults());
}

} // namespace mitigation
} // namespace catalyst
