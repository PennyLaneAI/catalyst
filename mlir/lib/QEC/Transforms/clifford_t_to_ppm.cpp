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

#define DEBUG_TYPE "ppm_compilation"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "QEC/IR/QECDialect.h"
#include "QEC/Transforms/Patterns.h"
#include "Quantum/IR/QuantumOps.h"
#include <algorithm>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::qec;

// bool qsimFilter(mlir::Operation *op) {
//     return op->getName().getStringRef() == "quantum.extract";
// }

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_CLIFFORDTTOPPMPASS
#define GEN_PASS_DECL_CLIFFORDTTOPPMPASS
#include "QEC/Transforms/Passes.h.inc"

struct CliffordTToPPMPass : public impl::CliffordTToPPMPassBase<CliffordTToPPMPass> {
    using CliffordTToPPMPassBase::CliffordTToPPMPassBase;

    void print_specs(StringRef pass_name)
    {
        llvm::outs() << pass_name <<"\n";
        llvm::BumpPtrAllocator string_allocator;
        llvm::DenseMap<StringRef, int> PPM_Specs;
        PPM_Specs["num_logical_qubits"] = 0;
        PPM_Specs["num_of_ppm"] = 0;

        // Walk over all operations in the IR (could be ModuleOp or FuncOp)
        getOperation()->walk([&](Operation *op) {
            // Skip top-level container ops if desired
            if (isa<ModuleOp>(op)) return;

            // llvm::outs()<<"\n-----------------------------MLIR------------------------------\n";
            // op->print(llvm::outs());
            // llvm::outs()<<"\n-----------------------------MLIR------------------------------\n";

            StringRef gate_name = op->getName().getStringRef();

            if (gate_name == "quantum.alloc") {
                auto num_qubits_attr = op->getAttrOfType<mlir::IntegerAttr>("nqubits_attr");
                u_int64_t num_qubits = num_qubits_attr ? static_cast<u_int64_t>(num_qubits_attr.getInt()) : 0;
                PPM_Specs["num_logical_qubits"] = num_qubits;
            }

            if (gate_name == "qec.ppm") {
                PPM_Specs["num_of_ppm"] = PPM_Specs["num_of_ppm"] + 1;
            }

            if (gate_name == "qec.ppr") {
                auto rotation_attr = op->getAttrOfType<mlir::IntegerAttr>("rotation_kind");
                auto pauli_product_attr = op->getAttrOfType<mlir::ArrayAttr>("pauli_product");
                int16_t rotation_kind = rotation_attr ? static_cast<int16_t>(rotation_attr.getInt()) : 0;
                if (rotation_kind) {
                    llvm::StringSaver saver(string_allocator);
                    StringRef num_pi_key = saver.save("num_pi"+std::to_string(abs(rotation_kind))+"_gates");
                    StringRef max_weight_pi_key = saver.save("max_weight_pi"+std::to_string(abs(rotation_kind)));

                    if (PPM_Specs.find(llvm::StringRef(num_pi_key)) == PPM_Specs.end()) {
                        PPM_Specs[num_pi_key] = 1;
                        PPM_Specs[max_weight_pi_key] = static_cast<int>(pauli_product_attr.size());
                    }
                    else {
                        PPM_Specs[num_pi_key] = PPM_Specs[num_pi_key] + 1;
                        PPM_Specs[max_weight_pi_key] = std::max(PPM_Specs[max_weight_pi_key], static_cast<int>(pauli_product_attr.size()));
                    }
                }
            }
            mlir::SetVector <Operation *> backwardSlice;
            getBackwardSlice(op, &backwardSlice);
            llvm::outs()<<"\n-----------------------------SLICE-----------------------------\n";
            llvm::outs()<<"Backward slicing\n";
            for (Operation *o : backwardSlice) {
                if (o->getName().getStringRef() == "quantum.extract") {
                    llvm::outs() << *o << "\n"; 
                }
            }
            llvm::outs()<<"\n-----------------------------SLICE------------------------------\n";
        });

        for (const auto &entry : PPM_Specs) {
            llvm::outs() << "  " << entry.first << ": " << entry.second << "\n";
        }
        llvm::outs()<<"\n=====================================================================\n";
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

        // Phase 2: Commute Clifford gates past T gates using PPR representation
        {
            RewritePatternSet patterns(ctx);
            populateCommuteCliffordTPPRPatterns(patterns, max_pauli_size);

            if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
                return signalPassFailure();
            }
            print_specs("Commute Clifford gates past T");
        }

        // Phase 3: Absorb Clifford gates into measurement operations
        {
            RewritePatternSet patterns(ctx);
            populateCommuteCliffordPastPPMPatterns(patterns, max_pauli_size);

            if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
                return signalPassFailure();
            }
            print_specs("Absorb Clifford gates into measurement operations");
        }

        // Phase 4: Decompose non-Clifford PPRs into PPMs
        {
            RewritePatternSet patterns(ctx);
            populateDecomposeNonCliffordPPRPatterns(patterns, decomposeMethod, avoidYMeasure);

            if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
                return signalPassFailure();
            }
            print_specs("Decompose non-Clifford PPRs into PPMs");
        }

        // Phase 5: Decompose Clifford PPRs into PPMs
        {
            RewritePatternSet patterns(ctx);
            populateDecomposeCliffordPPRPatterns(patterns, avoidYMeasure);

            if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
                return signalPassFailure();
            }
            print_specs("Decompose Clifford PPRs into PPMs");
        }
    }
};

} // namespace qec

std::unique_ptr<Pass> createCliffordTToPPMPass() { return std::make_unique<CliffordTToPPMPass>(); }

} // namespace catalyst
