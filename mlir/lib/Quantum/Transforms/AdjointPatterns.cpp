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


void replaceOperands(Operation &op,
                     ArrayRef<Value> templates,
                     ArrayRef<Value> replacement)
{
    assert(templates.size() == replacement.size());
    for (size_t i = 0 ; i < op.getNumOperands(); i++) {
        auto o = op.getOperand(i);
        auto res = std::find(templates.begin(), templates.end(), o);
        if (res == templates.end())
            continue;
        size_t res_i = res - templates.begin();
        op.setOperand(i, replacement[res_i]);
    }
}

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
        auto adjoint_operands = ({
            std::vector<Value> out;
            for(auto a : op.getOperands()) {
                LLVM_DEBUG(dbgs() << "adjoint operand: " << a << "\n");
                out.push_back(a);
            }
            out;
        });
        auto block_params = ({
            std::vector<Value> out;
            for(auto a : op.getRegion().front().getArguments()) {
                LLVM_DEBUG(dbgs() << "block param: " << a << "\n");
                out.push_back(a);
            }
            out;
        });

        assert(op.getRegion().hasOneBlock());
        Block &b = op.getRegion().front();
        auto rb = std::make_reverse_iterator(b.end());
        auto re = std::make_reverse_iterator(b.begin());

        llvm::DenseMap<Value,Value> reversal_mapping;

        for( auto i = rb; i!=re; i++) {
            LLVM_DEBUG(dbgs() << "reverse walking: " << i->getName() << " " << *i << "\n");
            if(YieldOp yield = isInstanceOf<YieldOp>(*i)) {
                LLVM_DEBUG(dbgs() << "yield! " << *yield << "\n");
                assert(yield.getOperands().size() == 1);
                auto qreg = ({
                    Value out = *yield.getResults().begin();
                    LLVM_DEBUG(dbgs() << "  qreg type: " << out.getType() << "\n");
                    out;
                    });
                reversal_mapping[qreg] = adjoint_operands[0];
            }
            else if(auto insert = isInstanceOf<InsertOp>(*i)) {
                auto extract = rewriter.create<ExtractOp>(
                    loc,
                    catalyst::quantum::QubitType(),
                    reversal_mapping[insert.getQubit()],
                    insert.getIdx(),
                    insert.getIdxAttrAttr()
                    );
                /* reversal_mapping[insert.getQubit()] = extract.getQubit(); */
            }
            else if(auto custom = isInstanceOf<CustomOp>(*i)) {
                auto customA = rewriter.create<CustomOp>(
                    loc,
                    custom.getResultTypes(),
                    custom.getParams(), ({
                        std::vector<Value> qbits;
                         for(auto q: custom.getInQubits()) {
                            qbits.push_back(reversal_mapping[q]);
                         }
                         qbits;
                    }),
                    custom.getGateName(),
                    mlir::BoolAttr::get(ctx, !custom.getAdjoint().value_or(false))
                );
                /* reversal_mapping[insert.getQubit()] = qbit; */
            }
            else if(auto extract = isInstanceOf<ExtractOp>(*i)) {
                auto insert = rewriter.create<InsertOp>(
                    loc,
                    catalyst::quantum::QuregType(),
                    extract.getResult(),
                    extract.getIdx(),
                    extract.getIdxAttrAttr(),
                    reversal_mapping[extract.getQubit()]
                    );
                /* reversal_mapping[insert.getQubit()] = insert.getOutQreg(); */
            }
            else {
            }
        }

        return failure();
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
