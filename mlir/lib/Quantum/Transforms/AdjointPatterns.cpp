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

#include <vector>
#include <iterator>
#include <algorithm>
#include <unordered_map>

#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace llvm;
using namespace mlir;
using namespace catalyst::quantum;

namespace {


template<class T>
T isInstanceOf(Operation &op)
{
  if(op.getName().getStringRef() == T::getOperationName())
      return cast<T>(op);
  else
      return nullptr;
}


struct AdjointSingleOpRewritePattern : public mlir::OpRewritePattern<AdjointOp> {
    using mlir::OpRewritePattern<AdjointOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(AdjointOp op, mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Adjointing the following:\n" << op << "\n");
        Location loc = op.getLoc();
        MLIRContext *ctx = op.getContext();
        assert(op.getRegion().hasOneBlock());

        // In essence, we build a map from values mentiond in the original program data flow to
        // the values of the program where quantum control flow is reversed. Most of the time,
        // there is a 1-to-1 correspondence with a notable exception caused by
        // `insert`/`extract` API asymetry.
        auto reversal_mapping = ({
            llvm::DenseMap<Value,Value> out;
            auto query = [&out] (Value key) -> Value {
                LLVM_DEBUG(dbgs() << "  querying: " << key << "\n");
                auto val = out[key];
                LLVM_DEBUG(dbgs() << "    result: " << val << "\n");
                return val;
            };
            auto update = [&out] (Value key, Value val) -> void {
                LLVM_DEBUG(dbgs() << "  updating: " << key << "\n");
                LLVM_DEBUG(dbgs() << "    to: " << val << "\n");
                out[key] = val;
            };
            Block &b = op.getRegion().front();
            auto rb = std::make_reverse_iterator(b.end());
            auto re = std::make_reverse_iterator(b.begin());
            for( auto i = rb; i!=re; i++) {
                LLVM_DEBUG(dbgs() << "operation: " << *i << "\n");
                if(YieldOp yield = isInstanceOf<YieldOp>(*i)) {
                    assert(yield.getOperands().size() == 1);
                    update(*yield.getResults().begin(), op.getQreg());
                }
                else if(InsertOp insert = isInstanceOf<InsertOp>(*i)) {
                    ExtractOp extract = rewriter.create<ExtractOp>(
                        loc,
                        insert.getQubit().getType(),
                        query(insert.getOutQreg()),
                        insert.getIdx(),
                        insert.getIdxAttrAttr()
                    );
                    update(insert.getQubit(), extract->getResult(0));
                    update(insert.getInQreg(), out[insert.getOutQreg()]);
                }
                else if(CustomOp custom = isInstanceOf<CustomOp>(*i)) {
                    assert(custom.getInQubits().size() == custom.getOutQubits().size() && "size");
                    auto in_qubits = ({
                        std::vector<Value> qbits;
                        for(auto q: custom.getOutQubits()) {
                            qbits.push_back(query(q));
                        }
                        qbits;
                    });
                    auto customA = rewriter.create<CustomOp>(
                        loc,
                        custom.getResultTypes(),
                        custom.getParams(),
                        in_qubits,
                        custom.getGateName(),
                        mlir::BoolAttr::get(ctx, !custom.getAdjoint().value_or(false))
                    );
                    for(size_t i = 0; i<customA.getOutQubits().size(); i++) {
                        update(custom.getInQubits()[i], customA->getResult(i));
                    }
                }
                else if(ExtractOp extract = isInstanceOf<ExtractOp>(*i)) {
                    auto insert = rewriter.create<InsertOp>(
                        loc,
                        extract.getQreg().getType(),
                        query(extract.getQreg()),
                        extract.getIdx(),
                        extract.getIdxAttrAttr(),
                        query(extract.getQubit())
                    );
                    update(extract.getQreg(), insert->getResult(0));
                }
                else {
                    /* TODO: We expect to handle Scf control flow instructions here. Stateless loops
                     * and conditionals should not be a problem. Stateful loops would probably
                     * require some kind of classical unrolling.
                     */
                    /* TODO: We also expect to handle a 1 level of nested adjoint blocks here as
                     * well. Just wiring their region contents to the output program should be
                     * enough. Handling arbitrary-deep nesting needs some investigation.
                     */
                }
            }
            out;
        });

        // Finally, we query the outputs of the reversed program using the input program's block
        // quantum arguments as keys.
        auto new_outputs = ({
            std::vector<Value> out;
            for(auto a : op.getRegion().front().getArguments()) {
                out.push_back(reversal_mapping[a]);
            }
            out;
        });

        rewriter.replaceOp(op, new_outputs);
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
