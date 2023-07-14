// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "adjoint"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace {

/// Clone the region of the adjoint operation `op` to the POI specified by the `rewriter`. Build and
/// return the value mapping `mapping`.
Value cloneAdjointRegion(AdjointOp op, PatternRewriter &rewriter, IRMapping &mapping)
{
    Block &b = op.getRegion().front();
    for (auto i = b.begin(); i != b.end(); i++) {
        if (YieldOp yield = dyn_cast<YieldOp>(*i)) {
            return mapping.lookupOrDefault(yield->getOperand(0));
        }
        else {
            rewriter.insert(i->clone(mapping));
        }
    }
    assert(false && "quantum.yield must present in the adjoint region");
}

struct AdjointSingleOpRewritePattern : public mlir::OpRewritePattern<AdjointOp> {
    using mlir::OpRewritePattern<AdjointOp>::OpRewritePattern;

    /// In essence, we build a map from values mentiond in the source data flow to the values of the
    /// program where quantum control flow is reversed. Most of the time, there is a 1-to-1
    /// correspondence with a notable exception caused by `insert`/`extract` API asymetry.
    mlir::LogicalResult matchAndRewrite(AdjointOp adjoint,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Adjointing the following:\n" << adjoint << "\n");
        Location loc = adjoint.getLoc();

        // First, copy the classical computations directly to the target POI and build the classical
        // value mapping dictionary.
        IRMapping classicalMapping;
        {
            Block &b = adjoint.getRegion().front();
            for (auto i = b.begin(); i != b.end(); i++) {
                if (!isa<QuantumDialect>(i->getDialect())) {
                    LLVM_DEBUG(dbgs() << "classical operation: " << *i << "\n");
                    rewriter.insert(i->clone(classicalMapping));
                }
            }
        };

        // Next, compute and copy the reversed quantum computation flow. The classical dependencies
        // such as gate parameters or qubit indices are already available in `classicalMapping`.
        llvm::DenseMap<Value, Value> quantumMapping;
        {
            auto query = [&quantumMapping](Value key) -> Value {
                LLVM_DEBUG(dbgs() << "  querying: " << key << "\n");
                auto val = quantumMapping[key];
                LLVM_DEBUG(dbgs() << "    result: " << val << "\n");
                return val;
            };
            auto update = [&quantumMapping](Value key, Value val) -> void {
                LLVM_DEBUG(dbgs() << "  updating: " << key << "\n");
                LLVM_DEBUG(dbgs() << "    to: " << val << "\n");
                quantumMapping[key] = val;
            };
            Block &b = adjoint.getRegion().front();
            auto rb = std::make_reverse_iterator(b.end());
            auto re = std::make_reverse_iterator(b.begin());
            for (auto i = rb; i != re; i++) {
                LLVM_DEBUG(dbgs() << "operation: " << *i << "\n");
                if (YieldOp yield = dyn_cast<YieldOp>(*i)) {
                    assert(yield.getOperands().size() == 1);
                    update(*yield.getResults().begin(), adjoint.getQreg());
                }
                else if (InsertOp insert = dyn_cast<InsertOp>(*i)) {
                    ExtractOp extract = rewriter.create<ExtractOp>(
                        loc, insert.getQubit().getType(), query(insert.getOutQreg()),
                        classicalMapping.lookupOrDefault(insert.getIdx()), insert.getIdxAttrAttr());
                    update(insert.getQubit(), extract->getResult(0));
                    update(insert.getInQreg(), quantumMapping[insert.getOutQreg()]);
                }
                else if (QuantumGate gate = dyn_cast<QuantumGate>(*i)) {
                    IRMapping m(classicalMapping);
                    for (const auto &[qr, qo] :
                         llvm::zip(gate.getQubitResults(), gate.getQubitOperands())) {
                        m.map(qo, query(qr));
                    }
                    QuantumGate clone = dyn_cast<QuantumGate>(rewriter.insert(i->clone(m)));
                    clone.setAdjointFlag(!gate.getAdjointFlag());
                    for (const auto &[qr, qo] :
                         llvm::zip(clone.getQubitResults(), gate.getQubitOperands())) {
                        update(qo, qr);
                    }
                }
                else if (ExtractOp extract = dyn_cast<ExtractOp>(*i)) {
                    auto insert = rewriter.create<InsertOp>(
                        loc, extract.getQreg().getType(), query(extract.getQreg()),
                        classicalMapping.lookupOrDefault(extract.getIdx()),
                        extract.getIdxAttrAttr(), query(extract.getQubit()));
                    update(extract.getQreg(), insert->getResult(0));
                }
                else if (AdjointOp adjoint2 = dyn_cast<AdjointOp>(*i)) {
                    IRMapping m(classicalMapping);
                    Block &b = adjoint2.getRegion().front();
                    for (const auto &[a, r] : llvm::zip(b.getArguments(), adjoint2->getResults())) {
                        m.map(a, query(r));
                    }
                    auto res = cloneAdjointRegion(adjoint2, rewriter, m);
                    update(adjoint2.getQreg(), res);
                }
                else {
                    /* TODO: We expect to handle Scf control flow instructions here. Stateless loops
                     * and conditionals should not be a problem. Stateful loops would probably
                     * require some kind of classical unrolling.
                     */
                }
            }
        };

        // Finally, query and return the quantum outputs of the reversed program using the known
        // input arguments of the source adjoint block as keys.
        std::vector<Value> reversedOutputs;
        {
            for (auto a : adjoint.getRegion().front().getArguments()) {
                reversedOutputs.push_back(quantumMapping[a]);
            }
        }

        rewriter.replaceOp(adjoint, reversedOutputs);
        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateAdjointPatterns(RewritePatternSet &patterns)
{
    patterns.add<AdjointSingleOpRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
