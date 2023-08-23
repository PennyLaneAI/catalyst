// Copyright 2022-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "GradMethods/Adjoint.hpp"
#include "GradMethods/FiniteDifference.hpp"
#include "GradMethods/HybridGradient.hpp"
#include "GradMethods/JVPVJPPatterns.hpp"
#include "GradMethods/ParameterShift.hpp"

#include "mlir/IR/PatternMatch.h"

#include "Gradient/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst::gradient;

namespace catalyst {
namespace gradient {

void populateLoweringPatterns(RewritePatternSet &patterns, StringRef lowerOnly, bool printActivity)
{
    patterns.add<HybridGradientLowering>(patterns.getContext(), printActivity);
    if (lowerOnly == "" || lowerOnly == "fd")
        patterns.add<FiniteDiffLowering>(patterns.getContext(), 1);
    if (lowerOnly == "" || lowerOnly == "ps")
        patterns.add<ParameterShiftLowering>(patterns.getContext());
    if (lowerOnly == "" || lowerOnly == "adj")
        patterns.add<AdjointLowering>(patterns.getContext(), 1);
    if (lowerOnly == "" || lowerOnly == "jp") {
        patterns.add<JVPLoweringPattern>(patterns.getContext());
        patterns.add<VJPLoweringPattern>(patterns.getContext());
    }
}

} // namespace gradient
} // namespace catalyst
