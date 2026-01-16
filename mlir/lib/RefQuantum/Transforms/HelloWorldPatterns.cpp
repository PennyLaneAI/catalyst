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

#define DEBUG_TYPE "rq-hello-world"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

// #include "RefQuantum/IR/RefQuantumOps.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "RefQuantum/Transforms/Patterns.h"

using namespace mlir;

namespace {

struct RQHelloWorldPattern : public OpRewritePattern<catalyst::quantum::CustomOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(catalyst::quantum::CustomOp op,
                                  PatternRewriter &rewriter) const override
    {
        llvm::errs() << "hello world! Visiting " << op << "\n";
        return success();
    }
};

} // namespace

namespace catalyst {
namespace ref_quantum {

void populateRQHelloWorldPatterns(RewritePatternSet &patterns)
{
    patterns.add<RQHelloWorldPattern>(patterns.getContext());
}

} // namespace ref_quantum
} // namespace catalyst
