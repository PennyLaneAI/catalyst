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

#define DEBUG_TYPE "cancel-inverse-ppr"

#include "QEC/IR/QECDialect.h"
#include "QEC/IR/QECOpInterfaces.h"
#include "QEC/Transforms/Patterns.h"
#include "QEC/Utils/PauliStringWrapper.h"

using namespace mlir;
using namespace catalyst::qec;

namespace {

} // namespace

namespace catalyst {
namespace qec {

void populateCancelInversesPatterns(RewritePatternSet &patterns)
{
    patterns.add<CancelInversePPR>(patterns.getContext(), maxPauliSize, 1);
}

} // namespace qec
} // namespace catalyst
