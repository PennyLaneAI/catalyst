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


#include "mlir/IR/PatternMatch.h"
#include "Quantum/IR/QuantumOps.h"

#include "Ion/Transforms/Patterns.h"


using namespace mlir;
using namespace catalyst::ion;
using namespace catalyst::quantum;

namespace catalyst {
namespace ion {

struct QuantumToIonRewritePattern : public mlir::OpRewritePattern<CustomOp> {
    using mlir::OpRewritePattern<CustomOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(CustomOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        // D: Add PP Builder
        // Create test with RX(0) RY(1) MS(0,1)
        // Custom builder for device with ions
        // Optional value on device: value.get(), apply formula, give to operation
        // RX case: PP(P1, P2)
        // RY case: PP(P1, P2)
        // MS case: PP(P1, P2, P3, P4, P5, P6)
        // Else fail
        // Pulses Memory effect?
        return success();
    }
};


void populateQuantumToIonPatterns(RewritePatternSet &patterns)
{
    patterns.add<QuantumToIonRewritePattern>(patterns.getContext());
}

} // namespace ion
} // namespace catalyst
