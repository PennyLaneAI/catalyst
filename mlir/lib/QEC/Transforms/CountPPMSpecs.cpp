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

#define DEBUG_TYPE "ppm_specs"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "llvm/Support/Debug.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"
#include "QEC/Utils/PauliStringWrapper.h"
#include <algorithm>

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::qec;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_COUNTPPMSPECSPASS
#define GEN_PASS_DECL_COUNTPPMSPECSPASS
#include "QEC/Transforms/Passes.h.inc"


struct CountPPMSpecsPass : public impl::CountPPMSpecsPassBase<CountPPMSpecsPass> {
    using CountPPMSpecsPassBase::CountPPMSpecsPassBase;

    void print_specs(StringRef pass_name)
    {
        llvm::outs() << pass_name <<"\n";
        llvm::DenseMap<StringRef, int> PPM_Specs;
        PPM_Specs["num_pi4_gates"] = 0;
        PPM_Specs["num_pi8_gates"] = 0;
        PPM_Specs["max_weight_pi4"] = 0;
        PPM_Specs["max_weight_pi8"] = 0;

        // Walk over all operations in the IR (could be ModuleOp or FuncOp)
        getOperation()->walk([&](Operation *op) {
            // Skip top-level container ops if desired
            if (isa<ModuleOp>(op)) return;

            StringRef gate_name = op->getName().getStringRef();
            if (gate_name != "qec.ppr" && gate_name != "qec.ppm") return;

            auto rotation_attr = op->getAttrOfType<mlir::IntegerAttr>("rotation_kind");
            int16_t rotation_kind = rotation_attr ? static_cast<int16_t>(rotation_attr.getInt()) : 0;
            if (rotation_kind) {
                auto pauli_product_attr = op->getAttrOfType<mlir::ArrayAttr>("pauli_product");
                if (rotation_kind == 4 || rotation_kind == -4) {
                    PPM_Specs["num_pi4_gates"] = PPM_Specs["num_pi4_gates"] + 1;
                    PPM_Specs["max_wight_pi4"] = std::max(PPM_Specs["max_wight_pi4"], static_cast<int>(pauli_product_attr.size()));
                }
                if (rotation_kind == 8 || rotation_kind == -8) {
                    PPM_Specs["num_pi8_gates"] = PPM_Specs["num_pi8_gates"] + 1;
                    PPM_Specs["max_wight_pi8"] = std::max(PPM_Specs["max_wight_pi8"], static_cast<int>(pauli_product_attr.size()));
                }
            }
        });

        for (const auto &entry : PPM_Specs) {
            llvm::outs() << "  " << entry.first << ": " << entry.second << "\n";
        }
        return;
    }

    void runOnOperation() final
    {
        auto ctx = &getContext();
        auto module = getOperation();

        // Phase 1: Convert Clifford+T to PPR representation
        {
            ConversionTarget target(*ctx);
            target.addIllegalDialect<quantum::QuantumDialect>();
            target.addLegalOp<quantum::InitializeOp, quantum::FinalizeOp>();
            target.addLegalOp<quantum::DeviceInitOp, quantum::DeviceReleaseOp>();
            target.addLegalOp<quantum::AllocOp, quantum::DeallocOp>();
            target.addLegalOp<quantum::InsertOp, quantum::ExtractOp>();
            target.addLegalDialect<qec::QECDialect>();

            RewritePatternSet patterns(ctx);
            populateCliffordTToPPRPatterns(patterns);

            if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
                return signalPassFailure();
            }
            print_specs("Clifford+T to PPR");
        }


    }
};

} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<Pass> createCountPPMSpecsPass() { return std::make_unique<CountPPMSpecsPass>(); }

} // namespace catalyst
