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

#include "OpenQASM/IR/OpenQASMDialect.h"
#include "OpenQASM/IR/OpenQASMOps.h"
#include "OpenQASM/Transforms/Passes.h"

using namespace mlir;
using namespace catalyst::openqasm;

namespace catalyst {
namespace openqasm {

#define GEN_PASS_DECL_OPENQASMTRANSLATIONPASS
#define GEN_PASS_DEF_OPENQASMTRANSLATIONPASS
#include "OpenQASM/Transforms/Passes.h.inc"

struct OpenQASMTranslationPass : impl::OpenQASMTranslationPassBase<OpenQASMTranslationPass> {
    using OpenQASMTranslationPassBase::OpenQASMTranslationPassBase;

    void runOnOperation() final
    {
        Operation *module = getOperation();

        // Output the OpenQASM header
        llvm::outs() << "OPENQASM 3.0;\n";
        llvm::outs() << "include \"stdgates.inc\";\n\n";

        // Traverse the module and output OpenQASM
        module->walk([&](Operation *op) {
            if (auto allocateOp = dyn_cast<openqasm::AllocateOp>(op)) {
                llvm::outs() << "qubit[" << allocateOp.getSize() << "] q;\n";
            }
            else if (auto ryOp = dyn_cast<openqasm::RYOp>(op)) {
                llvm::outs() << "ry(" << ryOp.getAngle().convertToDouble() << ") q["
                             << ryOp.getQubit() << "];\n";
            }
            else if (auto cnotOp = dyn_cast<openqasm::CNOTOp>(op)) {
                llvm::outs() << "cx q[" << cnotOp.getControl() << "], q[" << cnotOp.getTarget()
                             << "];\n";
            }
        });

        llvm::outs() << "\n------------------------------\n";
        llvm::outs() << "ORIGINAL IR:\n";
        llvm::outs() << "------------------------------\n";
        llvm::outs() << "\n";
    }
};

} // namespace openqasm

std::unique_ptr<Pass> createOpenQASMTranslationPass()
{
    return std::make_unique<openqasm::OpenQASMTranslationPass>();
}

} // namespace catalyst
