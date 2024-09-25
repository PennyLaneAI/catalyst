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

#define DEBUG_TYPE "propagatesimplestates"

#include "PropagateSimpleStatesAnalysis.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumOps.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_PROPAGATESIMPLESTATESPASS
#define GEN_PASS_DECL_PROPAGATESIMPLESTATESPASS
#include "Quantum/Transforms/Passes.h.inc"

struct PropagateSimpleStatesTesterPass
    : public impl::PropagateSimpleStatesTesterPassBase<PropagateSimpleStatesTesterPass> {
    using impl::PropagateSimpleStatesTesterPassBase<
        PropagateSimpleStatesTesterPass>::PropagateSimpleStatesTesterPassBase;

    bool canScheduleOn(RegisteredOperationName opInfo) const override
    {
        return opInfo.hasInterface<FunctionOpInterface>();
    }

    void runOnOperation() override
    {
        LLVM_DEBUG(dbgs() << "propagate simple states pass"
                          << "\n");

        func::FuncOp func = cast<func::FuncOp>(getOperation());
        if (func.getSymName() != FuncNameOpt) {
            // not the function to run the pass on
            return;
        }

        ///////////////////////////

        PropagateSimpleStatesAnalysis &pssa = getAnalysis<PropagateSimpleStatesAnalysis>();
        llvm::DenseMap<Value, QubitState> qubitValues = pssa.getQubitValues();

        // We emit them as operation remarks for testing
        for (auto it = qubitValues.begin(); it != qubitValues.end(); ++it) {
            it->first.getDefiningOp()->emitRemark(pssa.QubitState2String(it->second));
        }
    }
};

std::unique_ptr<Pass> createPropagateSimpleStatesTesterPass()
{
    return std::make_unique<PropagateSimpleStatesTesterPass>();
}

} // namespace catalyst
