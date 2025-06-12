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

#include <algorithm>
#include <string>

#include <nlohmann/json.hpp>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "Quantum/IR/QuantumOps.h"
#include "QEC/IR/QECDialect.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::qec;
using json = nlohmann::json;

namespace catalyst {
namespace qec {

#define GEN_PASS_DEF_COUNTPPMSPECSPASS
#define GEN_PASS_DECL_COUNTPPMSPECSPASS
#include "QEC/Transforms/Passes.h.inc"

struct CountPPMSpecsPass : public impl::CountPPMSpecsPassBase<CountPPMSpecsPass> {
    using CountPPMSpecsPassBase::CountPPMSpecsPassBase;

    void print_specs()
    {
        llvm::BumpPtrAllocator string_allocator;
        llvm::DenseMap<StringRef, int> PPM_Specs;
        PPM_Specs["num_logical_qubits"] = 0;
        PPM_Specs["num_of_ppm"] = 0;

        // Walk over all operations in the IR (could be ModuleOp or FuncOp)
        getOperation()->walk([&](Operation *op) {
            // Skip top-level container ops if desired
            if (isa<ModuleOp>(op))
                return;

            StringRef gate_name = op->getName().getStringRef();

            if (gate_name == "quantum.alloc") {
                uint64_t num_qubits = cast<quantum::AllocOp>(op).getNqubitsAttr().value_or(0); 
                assert(num_qubits != 0 && "PPM specs with dynamic number of qubits is not implemented");
                PPM_Specs["num_logical_qubits"] = num_qubits;
            }

            if (gate_name == "qec.ppm") {
                PPM_Specs["num_of_ppm"]++;
            }

            if (gate_name == "qec.ppr") {
                auto rotation_attr = op->getAttrOfType<mlir::IntegerAttr>("rotation_kind");
                auto pauli_product_attr = op->getAttrOfType<mlir::ArrayAttr>("pauli_product");
                int16_t rotation_kind =
                    rotation_attr ? static_cast<int16_t>(rotation_attr.getInt()) : 0;
                if (rotation_kind) {
                    llvm::StringSaver saver(string_allocator);
                    StringRef num_pi_key =
                        saver.save("num_pi" + std::to_string(abs(rotation_kind)) + "_gates");
                    StringRef max_weight_pi_key =
                        saver.save("max_weight_pi" + std::to_string(abs(rotation_kind)));

                    if (PPM_Specs.find(llvm::StringRef(num_pi_key)) == PPM_Specs.end()) {
                        PPM_Specs[num_pi_key] = 1;
                        PPM_Specs[max_weight_pi_key] = static_cast<int>(pauli_product_attr.size());
                    }
                    else {
                        PPM_Specs[num_pi_key]++;
                        PPM_Specs[max_weight_pi_key] =
                            std::max(PPM_Specs[max_weight_pi_key],
                                     static_cast<int>(pauli_product_attr.size()));
                    }
                }
            }
        });

        json PPM_Specs_Json = PPM_Specs;
        llvm::outs() << PPM_Specs_Json.dump(4) << "\n";
        return;
    }

    void runOnOperation() final { print_specs(); }
};

} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<Pass> createCountPPMSpecsPass() { return std::make_unique<CountPPMSpecsPass>(); }

} // namespace catalyst
