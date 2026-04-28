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
#include "Quantum/Utils/PythonFunction.h"

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_INSTANTIATEPYTHONDECOMPRULEPASS
#include "Quantum/Transforms/Passes.h.inc"

struct InstantiatePythonDecompRulePass
    : public impl::InstantiatePythonDecompRulePassBase<InstantiatePythonDecompRulePass> {
    using InstantiatePythonDecompRulePassBase::InstantiatePythonDecompRulePassBase;

    void runOnOperation() override
    {
        llvm::StringSet<> addedWords;

        llvm::errs() << "starting pass\n";
        mlir::ModuleOp module = cast<mlir::ModuleOp>(getOperation());

        module.walk([&](quantum::PauliRotOp op) {
            llvm::errs() << "getting pauli word\n";
            std::string pauliWord;
            for (auto pauliChar : op.getPauliProduct()) {
                pauliWord += cast<mlir::StringAttr>(pauliChar).getValue().str();
            }
            llvm::errs() << "got pauli word\n";

            if (addedWords.contains(pauliWord)) {
                return mlir::WalkResult::advance();
            }

            std::vector<PyArg> args;
            args.push_back(0.2);       // dynamic parameter, any value will work
            args.push_back(pauliWord); // static parameter, must be the correct value
            llvm::errs() << "pauli word arg: " << pauliWord << "\n";
            std::vector<int> wires;
            llvm::errs() << "wires arg: {";
            for (size_t i = 0; i < pauliWord.size(); i++) {
                llvm::errs() << i << ", ";
                wires.push_back(i); // dynamic parameters, any value
            }
            llvm::errs() << "}\n";

            auto outOp = get_op_from_python(module, "catalyst.utils.python_callbacks",
                                            "test_rot_to_ppr", args, wires);
            outOp->setName((outOp->getName() + "_" + llvm::StringRef(pauliWord)).str());
            module.push_back(outOp.release());
            return mlir::WalkResult::advance();
        });
    }
};

} // namespace quantum
} // namespace catalyst
