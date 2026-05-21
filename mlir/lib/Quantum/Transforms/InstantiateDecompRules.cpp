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

#include "llvm/ADT/StringSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/DecompCallbacks.h"
#include "Quantum/Transforms/DecompCallbacksLoader.h"

using namespace mlir;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_INSTANTIATEDECOMPRULESPASS
#include "Quantum/Transforms/Passes.h.inc"

struct InstantiateDecompRulesPass
    : public impl::InstantiateDecompRulesPassBase<InstantiateDecompRulesPass> {
    using InstantiateDecompRulesPassBase::InstantiateDecompRulesPassBase;

    LogicalResult initialize([[maybe_unused]] MLIRContext *context) override
    {
        // QPC::PyInterpreterGuard::ensure();
        return success();
    }

    void runOnOperation() override
    {
        mlir::ModuleOp module = cast<mlir::ModuleOp>(getOperation());

        llvm::SmallVector<quantum::PauliRotOp> pauliRotOps;
        module.walk([&](quantum::PauliRotOp op) { pauliRotOps.push_back(op); });

        if (pauliRotOps.empty()) {
            return; // nothing to lower — callback not required
        }

        auto *cb = getDecompCallback();
        if (!cb) {
            loadPythonCallbackPlugin();
            cb = getDecompCallback();
        }
        if (!cb) {
            module.emitError("graph-decomposition needs a PauliRot callback; "
                             "rebuild Catalyst with -DCATALYST_ENABLE_PYTHON_CALLBACKS=ON "
                             "or ensure libQuantumPythonCallbacks is available to the driver");
            return signalPassFailure();
        }

        llvm::StringSet<> addedWords;

        for (quantum::PauliRotOp pauliRot : pauliRotOps) {
            std::string pauliWord;
            for (auto pauliChar : pauliRot.getPauliProduct()) {
                pauliWord += cast<mlir::StringAttr>(pauliChar).getValue().str();
            }
            if (addedWords.contains(pauliWord)) {
                continue;
            }
            addedWords.insert(pauliWord);

            llvm::SmallVector<int> wires(pauliRot.getInQubits().size());
            std::iota(wires.begin(), wires.end(), 0);

            auto outOp = cb->lowerPauliRot(&getContext(), 0.2, pauliWord, wires);

            if (!outOp) {
                return signalPassFailure();
            }

            outOp->setName((outOp->getName() + "_" + pauliWord).str()); // unique name per pauliword
            outOp.get()->setAttr(
                "target_gate", mlir::StringAttr::get(module.getContext(), "PauliRot" + pauliWord));

            module.push_back(outOp.release());
        }
    }
};

} // namespace quantum
} // namespace catalyst
