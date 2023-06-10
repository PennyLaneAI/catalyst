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
        Location loc = op.getLoc();
        MLIRContext *ctx = op.getContext();
        LLVM_DEBUG(dbgs() << "matching the adjoing op" << "\n");

        assert(op.getRegion().hasOneBlock());

        auto reversal_mapping = ({
            llvm::DenseMap<Value,Value> out;
            Block &b = op.getRegion().front();
            auto rb = std::make_reverse_iterator(b.end());
            auto re = std::make_reverse_iterator(b.begin());
            for( auto i = rb; i!=re; i++) {
                LLVM_DEBUG(dbgs() << "reversing: " << i->getName() << " " << *i << "\n");
                if(YieldOp yield = isInstanceOf<YieldOp>(*i)) {
                    LLVM_DEBUG(dbgs() << "  yield! " << yield << "\n");
                    assert(yield.getOperands().size() == 1);
                    auto qreg = ({
                        Value out = *yield.getResults().begin();
                        LLVM_DEBUG(dbgs() << "  qreg type: " << out.getType() << "\n");
                        out;
                        });
                    out[qreg] = op.getQreg();
                }
                else if(InsertOp insert = isInstanceOf<InsertOp>(*i)) {
                    LLVM_DEBUG(dbgs() << "  insert! " << insert << "\n");
                    ExtractOp extract = rewriter.create<ExtractOp>(
                        loc,
                        insert.getQubit().getType(),
                        out[insert.getOutQreg()],
                        insert.getIdx(),
                        insert.getIdxAttrAttr()
                    );
                    out[insert.getQubit()] = extract->getResult(0);
                    out[insert.getInQreg()] = out[insert.getOutQreg()];
                    LLVM_DEBUG(dbgs() << "  done! " << "\n");
                }
                else if(CustomOp custom = isInstanceOf<CustomOp>(*i)) {
                    LLVM_DEBUG(dbgs() << "  custom! " << custom << "\n");
                    assert(custom.getInQubits().size() == custom.getOutQubits().size() && "size");
                    auto in_qubits = ({
                        std::vector<Value> qbits;
                        for(auto q: custom.getOutQubits()) {
                            LLVM_DEBUG(dbgs() << "  finding! " << q << "\n");
                            assert(out.find(q) != out.end() && "not found!");
                            qbits.push_back(out[q]);
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
                        out[custom.getInQubits()[i]] = customA->getResult(i);
                    }
                }
                else if(ExtractOp extract = isInstanceOf<ExtractOp>(*i)) {
                    LLVM_DEBUG(dbgs() << "  extract! " << extract << "\n");
                    LLVM_DEBUG(dbgs() << "  finding! " << extract.getQreg() << "\n");
                    assert(out.find(extract.getQreg()) != out.end() && "not found!");
                    LLVM_DEBUG(dbgs() << "  finding! " << extract.getQubit() << "\n");
                    assert(out.find(extract.getQubit()) != out.end() && "not found!");
                    auto insert = rewriter.create<InsertOp>(
                        loc,
                        extract.getQreg().getType(),
                        out[extract.getQreg()],
                        extract.getIdx(),
                        extract.getIdxAttrAttr(),
                        out[extract.getQubit()]
                    );
                    LLVM_DEBUG(dbgs() << "  done! " << "\n");
                    out[extract.getQreg()] = insert->getResult(0);
                }
                else {
                    /* skip */
                }
            }
            out;
        });

        auto new_outputs = ({
            std::vector<Value> out;
            for(auto a : op.getRegion().front().getArguments()) {
                out.push_back(reversal_mapping[a]);
            }
            out;
        });

        rewriter.replaceOp(op, new_outputs);
        LLVM_DEBUG(dbgs() << "Adjoint lowering complete" << "\n");
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
