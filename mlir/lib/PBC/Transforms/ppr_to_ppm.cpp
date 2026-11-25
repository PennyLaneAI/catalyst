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

#define DEBUG_TYPE "ppr-to-ppm"

#include <optional>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Catalyst/Analysis/ResourceEffect.h"
#include "PBC/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::pbc;

namespace catalyst {
namespace pbc {

#define GEN_PASS_DECL_PPRTOPPMPASS
#define GEN_PASS_DEF_PPRTOPPMPASS
#include "PBC/Transforms/Passes.h.inc"

struct PPRToPPMPass : public impl::PPRToPPMPassBase<PPRToPPMPass> {
    using PPRToPPMPassBase::PPRToPPMPassBase;

    void runOnOperation() final {
        auto ctx = &getContext();
        auto module = getOperation();

        RewritePatternSet non_clifford_patterns(ctx);
        populateDecomposeNonCliffordPPRPatterns(non_clifford_patterns, decomposeMethod,
                                                avoidYMeasure);

        if (failed(applyPatternsGreedily(module, std::move(non_clifford_patterns)))) {
            return signalPassFailure();
        }

        // Decompose Clifford PPRs into PPMs
        RewritePatternSet clifford_patterns(ctx);
        populateDecomposeCliffordPPRPatterns(clifford_patterns, avoidYMeasure);

        if (failed(applyPatternsGreedily(module, std::move(clifford_patterns)))) {
            return signalPassFailure();
        }
    }

    LogicalResult verifyResourceEffect(Location loc) const {
        if (decomposeMethod != DecomposeMethod::PauliCorrected || avoidYMeasure) {
            return emitError(loc, "resource estimation only supports ppr-to-ppm with "
                                  "decompose-method=pauli-corrected and without avoid-y-measure");
        }
        return success();
    }
};

/// A pi/8 PPR consumes a magic state and a pi/4 PPR an ancilla, each measured jointly with the PPR
/// qubits, then on its own, followed by a Pauli correction (`decomposePauliCorrectedPiOverEight`
/// and `emitPiOverFourDecomposition`).
static std::optional<Substitution> getPPRToPPMSubstitution(const OpKey &op, const Pass &) {
    if (op.name != "PPR-pi/8" && op.name != "PPR-pi/4") {
        return std::nullopt;
    }
    Substitution substitution{
        {{{"PPM", op.numWires + 1, 0}, 1}, {{"PPM", 1, 0}, 1}, {{"PPR-pi/2", op.numWires, 0}, 1}},
        {/*alloc=*/1, /*peak=*/1}};
    if (op.name == "PPR-pi/8") {
        substitution.ops.push_back({{"pbc.fabricate", 0, 0}, 1});
    }
    return substitution;
}

void registerPPRToPPMResourceEffect() {
    [[maybe_unused]] static const bool once = [] {
        ResourceEffectRegistry::get().add(
            "ppr-to-ppm",
            makeSubstitutionEffect(getPPRToPPMSubstitution, [](const Pass &pass, Location loc) {
                return static_cast<const PPRToPPMPass &>(pass).verifyResourceEffect(loc);
            }));
        return true;
    }();
}

} // namespace pbc
} // namespace catalyst
