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

// This is a function to display all the CustomOpData for debugging purposes.
static void printCustomOpData(const CustomOpData &op)
{
    llvm::outs() << "CustomOpData:\n";
    llvm::outs() << "  Gate Name: " << op.gateName << "\n";
    llvm::outs() << "  Adjoint: " << (op.adjoint ? "true" : "false") << "\n";
    llvm::outs() << "  Params: ";
    for (const auto &param : op.params) {
        if (param.type == Param::ParamType::Float)
            llvm::outs() << param.f << " ";
    }
    llvm::outs() << "\n";
    llvm::outs() << "  Wire Indices: ";
    for (unsigned idx : op.wireIndices)
        llvm::outs() << idx << " ";
    llvm::outs() << "\n";
    llvm::outs() << "  Control Qubit Indices: ";
    for (unsigned idx : op.ctrlQubitIndices)
        llvm::outs() << idx << " ";
    llvm::outs() << "\n";
    llvm::outs() << "  Control Values: ";
    for (int64_t val : op.ctrlValues)
        llvm::outs() << val << " ";
    llvm::outs() << "\n";
}

// This is a function to display the content of qregs_map for debugging purposes.
static void printQregsMap(const llvm::DenseMap<Value, unsigned> &qregs_map)
{
    llvm::outs() << "Qregs Map:\n";
    for (const auto &pair : qregs_map) {
        llvm::outs() << "  Value: " << pair.first << ", Index: " << pair.second << "\n";
    }
}

struct DecomposePass : public impl::DecomposePassBase<DecomposePass> {
    using impl::DecomposePassBase<DecomposePass>::DecomposePassBase;

    void runOnOperation() override
    {

        auto module = getOperation();

        llvm::DenseMap<Value, unsigned> qregs_map;

        module->walk([&](quantum::ExtractOp ex) {
            if (auto idx = ex.getIdxAttr())
                qregs_map[ex.getResult()] = (unsigned)idx.value();
            else
                ex.emitWarning("ExtractOp without index, skipping");
        });

        std::vector<CustomOpData> CollectedCustomOps;
        // TODO: this reserve is arbitrary to avoid reallocations.
        // We can adjust it based on expected number of CustomOps
        // if it turns out to be a performance bottleneck.
        CollectedCustomOps.reserve(32);

        module->walk([&](quantum::CustomOp op) {
            CustomOpData op_data;
            op_data.gateName = op.getGateName();
            op_data.adjoint = op.getAdjoint();

            op_data.params.reserve(op.getParams().size());
            op_data.ctrlQubitIndices.reserve(op.getInCtrlQubits().size());
            op_data.wireIndices.reserve(op.getInQubits().size());
            op_data.ctrlValues.reserve(op.getInCtrlValues().size());

            for (Value pv : op.getParams()) {
                if (auto p = extractParam(pv))
                    op_data.params.push_back(*p);
                else
                    op.emitWarning("Non-constant parameter skipped");
            }

            for (Value cq : op.getInCtrlQubits()) {
                if (auto it = qregs_map.find(cq); it != qregs_map.end())
                    op_data.ctrlQubitIndices.push_back(it->second);
                else
                    op.emitWarning("Control qubit not in map");
            }

            for (Value cv : op.getInCtrlValues()) {
                if (auto constOp = cv.getDefiningOp<arith::ConstantOp>())
                    if (auto ia = mlir::dyn_cast<IntegerAttr>(constOp.getValue()))
                        op_data.ctrlValues.push_back(ia.getValue().getSExtValue());
            }

            auto inQ = op.getInQubits();
            auto outQ = op.getOutQubits();
            assert(inQ.size() == outQ.size());

            for (size_t i = 0; i < inQ.size(); ++i) {
                auto inVal = inQ[i];
                auto outVal = outQ[i];
                auto it = qregs_map.find(inVal);
                if (it == qregs_map.end()) {
                    op.emitWarning("Input qubit not in map");
                    continue;
                }
                unsigned idx = it->second;
                op_data.wireIndices.push_back(idx);
                qregs_map[outVal] = idx;
            }

            CollectedCustomOps.emplace_back(std::move(op_data));
        });

        for (const auto &data : CollectedCustomOps) {
            printCustomOpData(data);
            llvm::outs() << "\n";
        }

        printQregsMap(qregs_map);

        RewritePatternSet patterns(&getContext());
        patterns.add<DecomposeCustomOp>(&getContext());

        if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
            return signalPassFailure();
        }
    }
};

std::unique_ptr<Pass> createDecomposePass() { return std::make_unique<DecomposePass>(); }

}