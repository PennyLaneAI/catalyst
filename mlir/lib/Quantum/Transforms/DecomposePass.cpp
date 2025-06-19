#define DEBUG_TYPE "decompose"
#include <iostream>

#include "Catalyst/IR/CatalystDialect.h"
#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include <complex>
#include <optional>

struct Param {
    // TODO: Handle more types of attributes as needed.
    // For now, we only handle float parameters.
    enum class ParamType { Float } type;
    double f;
};

struct CustomOpData {
    std::string gateName;
    bool adjoint;
    unsigned numCtrlQubits;
    unsigned numCtrlValues;
    std::vector<Param> params;
    std::vector<unsigned> wireIndices;
    std::vector<unsigned> ctrlQubitIndices;
    std::vector<int64_t> ctrlValues;
};

using namespace llvm;
using namespace mlir;
using namespace catalyst;

namespace catalyst {
#define GEN_PASS_DEF_DECOMPOSEPASS
#define GEN_PASS_DECL_DECOMPOSEPASS
#include "Quantum/Transforms/Passes.h.inc"

static std::optional<Param> extractParam(Value v)
{
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        Attribute attr = constOp.getValue();

        if (auto floatAttr = mlir::dyn_cast<FloatAttr>(attr)) {
            return Param{Param::ParamType::Float, floatAttr.getValueAsDouble()};
        }
    }
    return std::nullopt;
}

struct DecomposeCustomOp : public OpRewritePattern<quantum::CustomOp> {
    using mlir::OpRewritePattern<quantum::CustomOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(quantum::CustomOp op, PatternRewriter &rewriter) const override
    {
        // TODO: Implement the actual decomposition logic.
        return failure();
    }
};

struct DecomposePass : public impl::DecomposePassBase<DecomposePass> {
    using impl::DecomposePassBase<DecomposePass>::DecomposePassBase;

    void runOnOperation() override
    {
        llvm::DenseMap<mlir::Value, unsigned> qregs_map;
        std::vector<CustomOpData> CollectedCustomOps;
        CollectedCustomOps.reserve(32);

        auto module = getOperation();

        module->walk([&](quantum::ExtractOp extractOp) {
            auto IdxAttr = extractOp.getIdxAttr();
            if (!IdxAttr) {
                extractOp.emitWarning("ExtractOp without index attribute, skipping");
                return;
            }
            unsigned wireIndex = static_cast<unsigned>(IdxAttr.value());
            qregs_map[extractOp.getResult()] = wireIndex;
        });

        module->walk([&](quantum::CustomOp op) {
            CustomOpData collected_op;
            collected_op.gateName = op.getGateName();
            collected_op.adjoint = op.getAdjoint();
            collected_op.numCtrlQubits = op.getInCtrlQubits().size();
            collected_op.numCtrlValues = op.getInCtrlValues().size();
            CollectedCustomOps.push_back(collected_op);

            // Collect the parameters of the custom operation
            auto rawParams = op.getParams();
            collected_op.params.reserve(rawParams.size());

            for (auto pv : rawParams) {
                if (auto p = extractParam(pv))
                    collected_op.params.push_back(*p);
                else
                    op.emitWarning("Non-constant parameter skipped");
            }

            // Collect the control qubits
            for (auto cq : op.getInCtrlQubits()) {
                auto it = qregs_map.find(cq);
                if (it != qregs_map.end())
                    collected_op.ctrlQubitIndices.push_back(it->second);
            }

            // Collect the wires in the qregs_map
            auto inQubits = op.getInQubits();
            auto outQubits = op.getOutQubits();
            assert(inQubits.size() == outQubits.size());

            for (unsigned i = 0; i < inQubits.size(); ++i) {
                auto it = qregs_map.find(inQubits[i]);
                if (it == qregs_map.end())
                    continue;
                qregs_map[outQubits[i]] = it->second;
            }

            // Collect the wires (TODO: combine with above)
            for (auto q : op.getInQubits()) {
                auto it = qregs_map.find(q);
                if (it != qregs_map.end())
                    collected_op.wireIndices.push_back(it->second);
            }
        });

        RewritePatternSet patterns(&getContext());
        patterns.add<DecomposeCustomOp>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDecomposePass() { return std::make_unique<DecomposePass>(); }

} // namespace catalyst