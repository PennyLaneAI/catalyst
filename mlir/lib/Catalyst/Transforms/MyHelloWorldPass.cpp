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

#define DEBUG_TYPE "myhelloworld"

#include "Catalyst/IR/CatalystDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_MYHELLOWORLDPASS
#define GEN_PASS_DECL_MYHELLOWORLDPASS
#include "Catalyst/Transforms/Passes.h.inc"

struct MyHelloWorldPass : public impl::MyHelloWorldPassBase<MyHelloWorldPass> {
    using impl::MyHelloWorldPassBase<MyHelloWorldPass>::MyHelloWorldPassBase;

    void runOnOperation() override { llvm::errs() << "Hello world!\n"; }
};

std::unique_ptr<Pass> createMyHelloWorldPass() { return std::make_unique<MyHelloWorldPass>(); }

} // namespace catalyst
