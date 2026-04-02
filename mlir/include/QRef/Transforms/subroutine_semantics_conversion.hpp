// Copyright 2026 Xanadu Quantum Technologies Inc.

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


#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "QRef/IR/QRefOps.h"
#include "QRef/Transforms/value_semantics_conversion.h"

using namespace mlir;
using namespace catalyst;

namespace ReferenceToValueSemanticsConversion {

struct SubroutineInfo {

}; // struct SubroutineInfo


void getNecessarySubroutineRValues(func::FuncOp f, SetVector<Value> &necessarySubroutineRValues)
{
    auto *qrefDialect = f.getContext()->getLoadedDialect<qref::QRefDialect>();
    llvm::SmallDenseSet<Value, 8> rQregsTakenIn;

    f.walk([&](Operation *op) {
        if (op->getDialect() != qrefDialect && !isa<func::CallOp>(op)) {
            return;
        }
        if (isa<qref::GetOp>(op)) {
            return;
        }
        for (Value v : op->getOperands()) {
            if (isa<qref::QuregType>(v.getType())) {
                // Ignore allocations from inside the region itself
                if (llvm::is_contained(f.getArguments(), v)) {
                    necessarySubroutineRValues.insert(v);
                    rQregsTakenIn.insert(v);
                }
            }
            else if (isa<qref::QubitType>(v.getType())) {
                if (isa<BlockArgument>(v) || !isa<qref::GetOp>(v.getDefiningOp())) {
                    // Ignore allocations from inside the region itself
                    if (llvm::is_contained(f.getArguments(), v)) {
                        necessarySubroutineRValues.insert(v);
                    }
                }
                else {
                    Value rQreg = getRSourceRegisterValue(v);
                    if (llvm::is_contained(f.getArguments(), rQreg)) {
                        auto getOp = cast<qref::GetOp>(v.getDefiningOp());
                        if (getOp.getIdx()) {
                            // dynamic extract index, must take in the reg
                            necessarySubroutineRValues.insert(rQreg);
                            rQregsTakenIn.insert(rQreg);
                        }
                        else {
                            necessarySubroutineRValues.insert(v);
                        }
                    }
                }
            }
        }
    });

    // If any rQregs are taken in, any rQubits belonging to them must not be taken in separately
    necessarySubroutineRValues.remove_if([&](const Value &v) {
        if (isa<BlockArgument>(v)) {
            return false;
        }
        if (auto getOp = dyn_cast<qref::GetOp>(v.getDefiningOp())) {
            if (rQregsTakenIn.contains(getOp.getQreg())) {
                return true;
            }
        }
        return false;
    });

    // Remove aliasing get ops
    DenseSet<rQubitGetOpInfo> seenGetInfos;
    necessarySubroutineRValues.remove_if([&](const Value &v) {
        if (isa<BlockArgument>(v) || !isa<qref::GetOp>(v.getDefiningOp())) {
            return false;
        }

        rQubitGetOpInfo info = getGetOpInfo(v).value();
        // If already exists in set, insertion will fail, and we have seen an alias, so need to
        // remove
        return !seenGetInfos.insert(info).second;
    });
}
} // namespace ReferenceToValueSemanticsConversion
