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

#define DEBUG_TYPE "disentanglecnot"

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
#define GEN_PASS_DEF_DISENTANGLECNOTPASS
#define GEN_PASS_DECL_DISENTANGLECNOTPASS
#include "Quantum/Transforms/Passes.h.inc"

struct DisentangleCNOTPass : public impl::DisentangleCNOTPassBase<DisentangleCNOTPass> {
    using impl::DisentangleCNOTPassBase<DisentangleCNOTPass>::DisentangleCNOTPassBase;

    bool canScheduleOn(RegisteredOperationName opInfo) const override
    {
        return opInfo.hasInterface<FunctionOpInterface>();
    }

    void runOnOperation() override
    {
        LLVM_DEBUG(dbgs() << "disentangle CNOT pass\n");

        func::FuncOp func = cast<func::FuncOp>(getOperation());
        if (func.getSymName() != FuncNameOpt) {
            // not the function to run the pass on
            return;
        }

        ///////////////////////////

        PropagateSimpleStatesAnalysis &pssa = getAnalysis<PropagateSimpleStatesAnalysis>();
        llvm::DenseMap<Value, QubitState> qubitValues = pssa.getQubitValues();

        func->walk([&](quantum::CustomOp op) {
            StringRef gate = op.getGateName();
            if (gate != "CNOT") {
                return;
            }

            llvm::errs() << "visiting " << op << "!\n";

            Value control_in = op->getOperand(0);
            Value target_in = op->getOperand(1);
            Value control_out = op->getResult(0);
            Value target_out = op->getResult(1);

            // |0> control, always do nothing
            if (pssa.isZero(qubitValues[control_in])) {
                control_out.replaceAllUsesWith(control_in);
                target_out.replaceAllUsesWith(target_in);
                op->erase();
                return;
            }

            /*
            for (auto operand : op->getOperands()){
                llvm::errs() << operand << " : " << qubitValues.contains(operand) << "\n";
                if (qubitValues.contains(operand)){
                    llvm::errs() << QubitState2String(qubitValues[operand]) <<
                    pssa.isZero(qubitValues[operand]) << "\n";
                }
            }
            */
        });
    }
};

std::unique_ptr<Pass> createDisentangleCNOTPass()
{
    return std::make_unique<DisentangleCNOTPass>();
}

} // namespace catalyst
