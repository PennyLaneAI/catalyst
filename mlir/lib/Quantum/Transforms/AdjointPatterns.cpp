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

template <class T> T isInstanceOf(Operation &op)
{
    if (op.getName().getStringRef() == T::getOperationName())
        return cast<T>(op);
    else
        return nullptr;
}

/// Copy the region of the adjoint operation `op` to the POI specified by the `rewriter`. Build and
/// return the value mapping `bvm`.
Value copyAdjointVerbatim(AdjointOp op, PatternRewriter &rewriter, IRMapping &bvm)
{
    Block &b = op.getRegion().front();
    for (auto i = b.begin(); i != b.end(); i++) {
        if (YieldOp yield = isInstanceOf<YieldOp>(*i)) {
            assert(++i == b.end() &&
                   "quantum.yield must be the last operation of an adjoint block");
            return bvm.lookupOrDefault(yield->getOperand(0));
        }
        else {
            rewriter.insert(i->clone(bvm));
        }
    }
    assert(false && "quantum.yield must present in the adjoint region");
}

struct AdjointSingleOpRewritePattern : public mlir::OpRewritePattern<AdjointOp> {
    using mlir::OpRewritePattern<AdjointOp>::OpRewritePattern;

    /// In essence, we build a map from values mentiond in the source data flow to the values of the
    /// program where quantum control flow is reversed. Most of the time, there is a 1-to-1
    /// correspondence with a notable exception caused by `insert`/`extract` API asymetry.
    mlir::LogicalResult matchAndRewrite(AdjointOp op,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Adjointing the following:\n" << op << "\n");
        Location loc = op.getLoc();
        MLIRContext *ctx = op.getContext();
        assert(op.getRegion().hasOneBlock());

        // First, copy the classical computations directly to the target POI and build the classical
        // value mapping dictionary.
        auto classicalMapping = ({
            IRMapping out;
            Block &b = op.getRegion().front();
            for (auto i = b.begin(); i != b.end(); i++) {
                if (i->getName().getStringRef().find("quantum") != std::string::npos) {
                    /* Skip quantum operations */
                }
                else {
                    LLVM_DEBUG(dbgs() << "classical operation: " << *i << "\n");
                    rewriter.insert(i->clone(out));
                }
            }
            out;
        });

        // Next, compute and copy the reversed quantum computation flow. The classical dependencies
        // such as gate parameters or qubit indices are already available in `classicalMapping`.
        auto quantumMapping = ({
            llvm::DenseMap<Value, Value> out;
            auto query = [&out](Value key) -> Value {
                LLVM_DEBUG(dbgs() << "  querying: " << key << "\n");
                auto val = out[key];
                LLVM_DEBUG(dbgs() << "    result: " << val << "\n");
                return val;
            };
            auto update = [&out](Value key, Value val) -> void {
                LLVM_DEBUG(dbgs() << "  updating: " << key << "\n");
                LLVM_DEBUG(dbgs() << "    to: " << val << "\n");
                out[key] = val;
            };
            Block &b = op.getRegion().front();
            auto rb = std::make_reverse_iterator(b.end());
            auto re = std::make_reverse_iterator(b.begin());
            for (auto i = rb; i != re; i++) {
                LLVM_DEBUG(dbgs() << "operation: " << *i << "\n");
                if (YieldOp yield = isInstanceOf<YieldOp>(*i)) {
                    assert(yield.getOperands().size() == 1);
                    update(*yield.getResults().begin(), op.getQreg());
                }
                else if (InsertOp insert = isInstanceOf<InsertOp>(*i)) {
                    ExtractOp extract = rewriter.create<ExtractOp>(
                        loc, insert.getQubit().getType(), query(insert.getOutQreg()),
                        classicalMapping.lookupOrDefault(insert.getIdx()),
                        insert.getIdxAttrAttr());
                    update(insert.getQubit(), extract->getResult(0));
                    update(insert.getInQreg(), out[insert.getOutQreg()]);
                }
                else if (CustomOp custom = isInstanceOf<CustomOp>(*i)) {
                    assert(custom.getInQubits().size() == custom.getOutQubits().size() &&
                        "Quantum operation must have inputs and outputs of the same qubit number");
                    auto in_qubits = ({
                        std::vector<Value> qbits;
                        for (auto q : custom.getOutQubits()) {
                            qbits.push_back(query(q));
                        }
                        qbits;
                    });
                    auto in_params = ({
                        std::vector<Value> out;
                        for (auto p : custom.getParams()) {
                            out.push_back(classicalMapping.lookupOrDefault(p));
                        }
                        out;
                    });
                    auto customA = rewriter.create<CustomOp>(
                        loc, custom.getResultTypes(), in_params, in_qubits, custom.getGateName(),
                        mlir::BoolAttr::get(ctx, !custom.getAdjoint().value_or(false)));
                    for (size_t i = 0; i < customA.getOutQubits().size(); i++) {
                        update(custom.getInQubits()[i], customA->getResult(i));
                    }
                }
                else if (ExtractOp extract = isInstanceOf<ExtractOp>(*i)) {
                    auto insert = rewriter.create<InsertOp>(
                        loc, extract.getQreg().getType(), query(extract.getQreg()),
                        classicalMapping.lookupOrDefault(extract.getIdx()),
                        extract.getIdxAttrAttr(), query(extract.getQubit()));
                    update(extract.getQreg(), insert->getResult(0));
                }
                else if (AdjointOp adjoint = isInstanceOf<AdjointOp>(*i)) {
                    IRMapping bvm(classicalMapping);
                    assert(adjoint.getRegion().hasOneBlock());
                    Block &b = adjoint.getRegion().front();
                    for (const auto &[a, r] : llvm::zip(b.getArguments(), adjoint->getResults())) {
                        bvm.map(a, query(r));
                    }
                    auto res = copyAdjointVerbatim(adjoint, rewriter, bvm);
                    update(adjoint.getQreg(), res);
                }
                else {
                    /* TODO: We expect to handle Scf control flow instructions here. Stateless loops
                     * and conditionals should not be a problem. Stateful loops would probably
                     * require some kind of classical unrolling.
                     */
                }
            }
            out;
        });

        // Finally, query and return the quantum outputs of the reversed program using the known
        // input arguments of the source adjoint block as keys.
        auto reversedOutputs = ({
            std::vector<Value> out;
            for (auto a : op.getRegion().front().getArguments()) {
                out.push_back(quantumMapping[a]);
            }
            out;
        });

        rewriter.replaceOp(op, reversedOutputs);
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
