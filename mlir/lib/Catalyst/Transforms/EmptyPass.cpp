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

// This pass does nothing.«

#define DEBUG_TYPE "empty"

#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"

using namespace llvm;
using namespace mlir;

namespace catalyst {

#define GEN_PASS_DECL_EMPTYPASS
#define GEN_PASS_DEF_EMPTYPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct EmptyPass : impl::EmptyPassBase<EmptyPass> {
    using EmptyPassBase::EmptyPassBase;

    void runOnOperation() final {
        LLVM_DEBUG(dbgs() << "empty pass"
                          << "\n");
        return;
    }
};

} // namespace catalyst
