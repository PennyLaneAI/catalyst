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

#define DEBUG_TYPE "PPMSpecs"

#include <algorithm>
#include <string>

#include <nlohmann/json.hpp>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "QEC/IR/QECDialect.h"
#include "Quantum/IR/QuantumOps.h"

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

    llvm::DenseMap<StringRef, int> countLogicalQubit(Operation *op,
                                                     llvm::DenseMap<StringRef, int> PPMSpecs)
    {
        uint64_t numQubits = cast<quantum::AllocOp>(op).getNqubitsAttr().value_or(0);
        assert(numQubits != 0 && "PPM specs with dynamic number of qubits is not implemented");
        PPMSpecs["num_logical_qubits"] = numQubits;
        return PPMSpecs;
    }
    llvm::DenseMap<StringRef, int> countPPR(Operation *op, llvm::DenseMap<StringRef, int> PPMSpecs,
                                            llvm::BumpPtrAllocator *stringAllocator)
    {
        int16_t rotationKind =
            cast<qec::PPRotationOp>(op).getRotationKindAttr().getValue().getZExtValue();
        auto PauliProductAttr = cast<qec::PPRotationOp>(op).getPauliProductAttr();
        if (rotationKind) {
            llvm::StringSaver saver(*stringAllocator);
            StringRef numRotationKindKey =
                saver.save("num_pi" + std::to_string(abs(rotationKind)) + "_gates");
            StringRef maxWeightRotationKindKey =
                saver.save("max_weight_pi" + std::to_string(abs(rotationKind)));

            if (PPMSpecs.find(llvm::StringRef(numRotationKindKey)) == PPMSpecs.end()) {
                PPMSpecs[numRotationKindKey] = 1;
                PPMSpecs[maxWeightRotationKindKey] = static_cast<int>(PauliProductAttr.size());
            }
            else {
                PPMSpecs[numRotationKindKey]++;
                PPMSpecs[maxWeightRotationKindKey] = std::max(
                    PPMSpecs[maxWeightRotationKindKey], static_cast<int>(PauliProductAttr.size()));
            }
        }
        return PPMSpecs;
    }
    void printSpecs()
    {
        llvm::BumpPtrAllocator stringAllocator;
        llvm::DenseMap<StringRef, int> PPMSpecs;
        PPMSpecs["num_logical_qubits"] = 0;
        PPMSpecs["num_of_ppm"] = 0;

        // Walk over all operations in the IR (could be ModuleOp or FuncOp)
        getOperation()->walk([&](Operation *op) {
            // Skip top-level container ops if desired
            if (isa<ModuleOp>(op))
                return;

            if (isa<quantum::AllocOp>(op)) {
                PPMSpecs = countLogicalQubit(op, PPMSpecs);
            }

            else if (isa<qec::PPMeasurementOp>(op)) {
                PPMSpecs["num_of_ppm"]++;
            }

            else if (isa<qec::PPRotationOp>(op)) {
                PPMSpecs = countPPR(op, PPMSpecs, &stringAllocator);
            }
        });

        json PPMSpecsJson = PPMSpecs;
        llvm::outs() << PPMSpecsJson.dump(4)
                     << "\n"; // dump(4) makes an indent with 4 spaces when printing JSON
        return;
    }

    void runOnOperation() final { printSpecs(); }
};

} // namespace qec

/// Create a pass for lowering operations in the `QECDialect`.
std::unique_ptr<Pass> createCountPPMSpecsPass() { return std::make_unique<CountPPMSpecsPass>(); }

} // namespace catalyst
