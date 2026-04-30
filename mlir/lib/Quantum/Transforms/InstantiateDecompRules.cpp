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
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumOps.h"
#include "QuantumPythonCallbacks/PythonFunction.hpp"

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_INSTANTIATEDECOMPRULESPASS
#include "Quantum/Transforms/Passes.h.inc"

struct InstantiateDecompRulesPass
    : public impl::InstantiateDecompRulesPassBase<InstantiateDecompRulesPass> {
    using InstantiateDecompRulesPassBase::InstantiateDecompRulesPassBase;

    void runOnOperation() override
    {
        mlir::ModuleOp module = cast<mlir::ModuleOp>(getOperation());

        llvm::StringSet<> addedWords;

        llvm::SmallVector<quantum::PauliRotOp> pauliRotOps;
        module.walk([&](quantum::PauliRotOp op) { pauliRotOps.push_back(op); });

        for (quantum::PauliRotOp pauliRot : pauliRotOps) {
            std::string pauliWord;
            for (auto pauliChar : pauliRot.getPauliProduct()) {
                pauliWord += cast<mlir::StringAttr>(pauliChar).getValue().str();
            }
            if (addedWords.contains(pauliWord)) {
                continue;
            }
            addedWords.insert(pauliWord);

            std::vector<PyArg> args;
            args.push_back(0.2);       // dynamic parameter, any value will work
            args.push_back(pauliWord); // static parameter, must be the correct value
            PyWires wires(pauliRot.getInQubits().size());
            std::iota(wires.begin(), wires.end(), 0);

            mlir::OwningOpRef<mlir::func::FuncOp> outOp =
                lowerPauliRotDecomp(module, "pennylane.ops.qubit.parametric_ops_multi_qubit",
                                    "_pauli_rot_decomposition", args, wires);

            outOp->setName(
                (outOp->getName() + "_" + pauliWord).str()); // unique name for decomp rule
            outOp.get()->setAttr(
                "target_gate", mlir::StringAttr::get(module.getContext(), "PauliRot" + pauliWord));

            module.push_back(outOp.release());
        }
    }
};

} // namespace quantum
} // namespace catalyst
