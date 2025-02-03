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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/TableGen/Record.h"
#include <vector>
#define DEBUG_TYPE "loop-boundary"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using llvm::dbgs;
using namespace llvm;
using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

static const mlir::StringSet<> rotationsSet = {"RX",  "RY",  "RZ",  "PhaseShift",
                                               "CRX", "CRY", "CRZ", "ControlledPhaseShift"};

static const mlir::StringSet<> hamiltonianSet = {"H", "X", "Y", "Z"};

static const mlir::StringSet<> multiQubitSet = {"CNOT", "CZ", "SWAP"};

namespace {

// TODO: move to a separate file
//===----------------------------------------------------------------------===//
//                        Helper functions
//===----------------------------------------------------------------------===//

// Represents the origin of a qubit, including its source register,
// position within the register, and whether it's part of a register.
struct QubitOrigin {
    mlir::Value sourceRegister;
    unsigned long position;
    bool isRegister;

    QubitOrigin()
        : sourceRegister(nullptr), position(0), isRegister(false) {}

    QubitOrigin(mlir::Value reg, unsigned long pos, bool isReg)
        : sourceRegister(reg), position(pos), isRegister(isReg) {}

    bool operator==(const QubitOrigin &other) const {
        return sourceRegister == other.sourceRegister && position == other.position;
    }
};

// Holds information about a quantum operation, 
// including its input qubits and parameters.
struct QuantumOpInfo {

    CustomOp op;
    std::vector<QubitOrigin> inQubits; // Control qubits
    bool isTopEdge;                    // Top edge flag

    QuantumOpInfo(CustomOp op, std::vector<QubitOrigin> inQubits, bool isTop)
        : op(op), inQubits(std::move(inQubits)), isTopEdge(isTop) {}
};

// map the operation to the list of qubit origins
template <typename OpType> using QubitOriginMap = std::map<OpType, std::vector<QubitOrigin>>;

// Checks if the given operation is a valid quantum operation based on its gate name.
bool isValidQuantumOperation(CustomOp &op)
{
    auto gateName = op.getGateName();
    return hamiltonianSet.contains(gateName) || 
           rotationsSet.contains(gateName) ||
           multiQubitSet.contains(gateName);
}

// Determines if the given operation has any successors that are quantum CustomOps.
bool hasQuantumCustomSuccessor(const CustomOp &op)
{
    return llvm::any_of(op->getUsers(), [](Operation *user) { 
        return isa<CustomOp>(user); 
    });
}

// Verifies that two sets of qubit origins are equivalent.
bool verifyQubitOrigins(const std::vector<QubitOrigin> &topOrigins,
                        const std::vector<QubitOrigin> &bottomOrigins)
{
    if (topOrigins.size() != bottomOrigins.size())
        return false;

    return std::equal(topOrigins.begin(), topOrigins.end(), bottomOrigins.begin());
}

template <typename OpType>
std::vector<std::pair<QuantumOpInfo, QuantumOpInfo>>
getVerifyEdgeOperationSet(QubitOriginMap<OpType> bottomEdgeOpSet,
                          QubitOriginMap<OpType> topEdgeOpSet)
{

    std::vector<std::pair<QuantumOpInfo, QuantumOpInfo>> edgeOperationSet;

    for (auto &[bottomOp, bottomQubitOrigins] : bottomEdgeOpSet) {

        for (auto &[topOp, topQubitOrigins] : topEdgeOpSet) {
            CustomOp bottomOpNonConst = bottomOp;
            CustomOp topOpNonConst = topOp;

            // Early rejection checks
            if (bottomOpNonConst.getGateName() != topOpNonConst.getGateName() ||
                !isValidQuantumOperation(bottomOpNonConst) || topOp == bottomOp ||
                !verifyQubitOrigins(topQubitOrigins, bottomQubitOrigins) ||
                hasQuantumCustomSuccessor(bottomOpNonConst)) {
                continue;
            }

            // Check if the qubits are the same
            bool hasQuantumPredecessor = false;

            // Check if the predecessor of the top op is not quantumCustom
            for (auto operand : topOpNonConst.getInQubits()) {
                if (auto op = operand.getDefiningOp()) {
                    if (dyn_cast_or_null<CustomOp>(op)) {
                        hasQuantumPredecessor = true;
                        break;
                    }
                }
            }

            if (hasQuantumPredecessor) {
                continue;
            }

            QuantumOpInfo topOpInfo = QuantumOpInfo(topOp, topQubitOrigins, true);
            QuantumOpInfo bottomOpInfo = QuantumOpInfo(bottomOp, bottomQubitOrigins, false);
            edgeOperationSet.push_back(std::make_pair(topOpInfo, bottomOpInfo));
        }
    }

    return edgeOperationSet;
}

template <typename OpType>
OpType createVectorOp(mlir::Value qreg, QubitOrigin qubit, mlir::PatternRewriter &rewriter,
                      mlir::Value element = nullptr)
{
    auto loc = qubit.sourceRegister.getLoc();
    auto integerAttr = rewriter.getI64IntegerAttr(qubit.position);

    if constexpr (std::is_same_v<OpType, quantum::InsertOp>) {
        assert(element && "InsertOp requires an element value!");
        return rewriter.create<quantum::InsertOp>(loc, qreg.getType(), qreg, nullptr, integerAttr,
                                                  element);
    }
    else if constexpr (std::is_same_v<OpType, quantum::ExtractOp>) {
        auto type = rewriter.getType<::catalyst::quantum::QubitType>();
        return rewriter.create<quantum::ExtractOp>(loc, type, qreg, nullptr, integerAttr);
    }
    else {
        static_assert(sizeof(OpType) == 0, "Unsupported operation type!");
    }
}

mlir::Value findInitValue(scf::ForOp forOp, mlir::Value qReg)
{
    for (auto [arg, regionArg] : llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
        if (qReg == regionArg) {
            return arg;
        }
    }
    return nullptr;
}

void updateForLoopInitArg(mlir::scf::ForOp forOp, mlir::Value sourceRegister, mlir::Value newValue,
                          mlir::PatternRewriter &rewriter)
{
    for (auto [idx, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        if (arg == sourceRegister) {
            auto index = forOp.getNumControlOperands() + idx;
            forOp.setOperand(index, newValue);
            break;
        }
    }
}

mlir::Value findResultForOpByQubit(mlir::Value qubit, scf::ForOp forOp)
{
    mlir::Value result;
    for (auto [idx, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        if (arg == qubit) {
            result = forOp.getResult(idx);
            break;
        }
    }
    return result;
}

//===----------------------------------------------------------------------===//
//                        Hoist functions
//===----------------------------------------------------------------------===//

// Traces the origin of qubits for the given operation and populates the qubitOriginMap.
template <typename OpType> void traceOriginQubit(OpType &op, QubitOriginMap<OpType> &qubitOriginMap)
{
    if (qubitOriginMap.count(op)) {
        return;
    }

    for (auto qubit : op.getInQubits()) {
        mlir::Value rootInQubit = qubit;
        while (auto definingOp = dyn_cast_or_null<OpType>(rootInQubit.getDefiningOp())) {
            for (auto [idx, outQubit] : llvm::enumerate(definingOp.getOutQubits())) {
                if (outQubit == rootInQubit) {
                    rootInQubit = definingOp.getInQubits()[idx];
                }
            }
        }

        if (ExtractOp extractOp =
                dyn_cast_or_null<quantum::ExtractOp>(rootInQubit.getDefiningOp())) {
            unsigned long long position = extractOp.getIdxAttr().value();
            QubitOrigin qubitOrigin = QubitOrigin(extractOp.getQreg(), position, true);
            qubitOriginMap[op].push_back(qubitOrigin);
            continue;
        }

        QubitOrigin qubitOrigin = QubitOrigin(rootInQubit, 0, false);
        qubitOriginMap[op].push_back(qubitOrigin);
    }
}

template <typename OpType> QubitOriginMap<OpType> traceTopEdgeOperations(scf::ForOp forOp)
{
    auto initArgs = forOp.getInitArgs();
    auto regionIterArgs = forOp.getRegionIterArgs();
    QubitOriginMap<OpType> qubitOriginMap;

    for (auto [arg, regionArg] : llvm::zip(initArgs, regionIterArgs)) {
        mlir::Type argType = arg.getType();

        for (Operation *op : regionArg.getUsers()) {

            if (isa<quantum::QuregType>(argType)) {
                if (ExtractOp extractOp = dyn_cast_or_null<quantum::ExtractOp>(op)) {
                    for (Operation *extractOpUser : extractOp->getUsers()) {
                        if (auto quantumOp = dyn_cast_or_null<OpType>(extractOpUser)) {
                            unsigned long long position = extractOp.getIdxAttr().value();
                            QubitOrigin qubitOrigin = QubitOrigin(regionArg, position, true);
                            
                            unsigned long index = 0;
                            for (auto operand : quantumOp.getInQubits()) {
                                if (operand == extractOp.getResult()) {
                                    break;
                                }
                                index++;
                            }

                            if (!qubitOriginMap.count(quantumOp) && qubitOriginMap[quantumOp].size() <= index) {
                                qubitOriginMap[quantumOp].resize(quantumOp.getInQubits().size());
                            }
                            qubitOriginMap[quantumOp][index] = qubitOrigin;
                        }
                    }
                }
                
            } else if (isa<quantum::QubitType>(argType)) {
                if (auto quantumOp = dyn_cast_or_null<OpType>(op)) {
                    QubitOrigin qubitOrigin = QubitOrigin(regionArg, 0, false);
                    qubitOriginMap[quantumOp].push_back(qubitOrigin);
                }
            }
        }
    }
    return qubitOriginMap;
}

template <typename OpType> QubitOriginMap<OpType> traceBottomEdgeOperations(scf::ForOp forOp)
{
    QubitOriginMap<OpType> qubitOriginMap;
    Operation *terminator = forOp.getBody()->getTerminator();

    for (auto operand : terminator->getOperands()) {

        Operation *definingOp = operand.getDefiningOp();

        // Handle the insert operation. e.g: %8 = quantum.insert %7[1], %6#1
        while (InsertOp insertOp = dyn_cast_or_null<quantum::InsertOp>(definingOp)) {
            // value of %6#1

            operand = insertOp.getQubit();

            // Get the operation of %6#1
            auto operation = dyn_cast_or_null<OpType>(operand.getDefiningOp());
            if (!operation) {
                continue;
            }

            // Trace the operation of %7 to find the origin of %7
            traceOriginQubit(operation, qubitOriginMap);

            // Get the operation of %7
            definingOp = insertOp.getInQreg().getDefiningOp();
        }

        // Check if the operation is not quantum
        auto operation = dyn_cast_or_null<OpType>(operand.getDefiningOp());

        if (!operation) {
            continue;
        }

        traceOriginQubit(operation, qubitOriginMap);
    }

    return qubitOriginMap;
}

template <typename OpType>
void hoistTopEdgeOperation(QuantumOpInfo topOpInfo, scf::ForOp forOp,
                           mlir::PatternRewriter &rewriter)
{
    rewriter.moveOpBefore(topOpInfo.op, forOp);
    topOpInfo.op.getOutQubits().replaceAllUsesWith(topOpInfo.op.getInQubits()); // config successor

    // hoist the extract operations
    std::vector<mlir::Value> operandOps;
    for (auto qubit : topOpInfo.inQubits) {
        mlir::Value initValue = findInitValue(forOp, qubit.sourceRegister);
        if (!initValue) {
            assert(false && "Register not found in loop arguments");
        }

        if (qubit.isRegister) {
            auto extractOp = createVectorOp<quantum::ExtractOp>(initValue, qubit, rewriter);
            rewriter.moveOpBefore(extractOp, topOpInfo.op);
            operandOps.push_back(extractOp.getResult());
        }
        else {
            operandOps.push_back(initValue);
        }
    }

    // Update op's operands
    // replace the operands of the topOpInfo.op with the operandOps
    for (auto [idx, operand] : llvm::enumerate(operandOps)) {
        unsigned long index = topOpInfo.op.getParams().size() + idx;
        topOpInfo.op.setOperand(index, operand);
    }

    // Move any tensor.extract operations that feed into topOp's inputs
    for (auto [idx, param] : llvm::enumerate(topOpInfo.op.getParams())) {
        if (auto extractOp = dyn_cast_or_null<tensor::ExtractOp>(param.getDefiningOp())) {
            auto extractOpClone = extractOp.clone();
            extractOpClone->setOperands(extractOp.getOperands());
            topOpInfo.op.setOperand(idx, extractOpClone.getResult());
            rewriter.setInsertionPoint(topOpInfo.op);
            rewriter.insert(extractOpClone);
        }
        else if (auto invarOp = findInitValue(forOp, param)) {
            topOpInfo.op.setOperand(idx, invarOp);
        }
        else {
            topOpInfo.op.setOperand(idx, param);
        }
    }

    // Create insert operations for the topOpInfo.op
    for (auto [idx, qubit] : llvm::enumerate(topOpInfo.inQubits)) {
        if (qubit.isRegister) {
            mlir::Value reg = findInitValue(forOp, qubit.sourceRegister);
            auto insertOp = createVectorOp<quantum::InsertOp>(reg, qubit, rewriter,
                                                              topOpInfo.op.getOutQubits()[idx]);
            // Find the matching init arg index and update it
            updateForLoopInitArg(forOp, qubit.sourceRegister, insertOp.getResult(), rewriter);
            rewriter.moveOpBefore(insertOp, forOp);
        }
        else {
            // if it is not a register, it is a qubit. so don't need to create vector operation
            // just need to find the matching init arg index and update it
            updateForLoopInitArg(forOp, qubit.sourceRegister, topOpInfo.op.getOutQubits()[idx],
                                 rewriter);
        }
    }
};

template <typename OpType>
void hoistBottomEdgeOperation(QuantumOpInfo bottomOpInfo, scf::ForOp forOp,
                              QubitOriginMap<CustomOp> bottomEdgeOpSet,
                              mlir::PatternRewriter &rewriter)
{

    // Config the successor
    bottomOpInfo.op.getOutQubits().replaceAllUsesWith(bottomOpInfo.op.getInQubits());

    // Move the operation after the for loop
    rewriter.moveOpAfter(bottomOpInfo.op, forOp);

    // Move any tensor.extract operations that feed into topOp's inputs
    for (auto [idx, param] : llvm::enumerate(bottomOpInfo.op.getParams())) {
        if (auto op = dyn_cast_or_null<mlir::Operation *>(param.getDefiningOp())) {
            auto opClone = op->clone();
            opClone->setOperands(op->getOperands());
            bottomOpInfo.op.setOperand(idx, opClone->getResult(0));
            rewriter.setInsertionPoint(bottomOpInfo.op);
            rewriter.insert(opClone);
        }
        else if (auto invarOp = findInitValue(forOp, param)) {
            bottomOpInfo.op.setOperand(idx, invarOp);
        }
        else {
            auto index = bottomOpInfo.op.getParams().size() + idx;
            bottomOpInfo.op.setOperand(index, param);
        }
    }

    // Create the insert operations
    for (auto [idx, qubit] : llvm::enumerate(bottomOpInfo.inQubits)) {

        mlir::Value reg;
        reg = findResultForOpByQubit(qubit.sourceRegister, forOp);

        auto outQubit = bottomOpInfo.op.getOutQubits()[idx];
        if (qubit.isRegister) {
            auto insertOp = createVectorOp<quantum::InsertOp>(reg, qubit, rewriter, outQubit);
            rewriter.moveOpAfter(insertOp, bottomOpInfo.op);
            reg.replaceAllUsesExcept(insertOp.getResult(), insertOp);
        }
        else {
            reg.replaceAllUsesExcept(outQubit, bottomOpInfo.op);
        }
    }

    // Create the extract operations
    std::vector<mlir::Value> operandOps;
    for (auto qubit : bottomOpInfo.inQubits) {
        // inint value should be forOp result
        mlir::Value reg = findResultForOpByQubit(qubit.sourceRegister, forOp);

        if (!reg) {
            assert(false && "Register not found in loop arguments");
        }
        if (qubit.isRegister) {
            auto extractOp = createVectorOp<quantum::ExtractOp>(reg, qubit, rewriter);
            rewriter.moveOpBefore(extractOp, bottomOpInfo.op);
            operandOps.push_back(extractOp.getResult());
        }
        else {
            operandOps.push_back(reg);
        }
    }

    // Replace the operands of the bottomOpInfo.op with the operandOps
    for (auto [idx, operand] : llvm::enumerate(operandOps)) {
        unsigned long index = bottomOpInfo.op.getParams().size() + idx;
        bottomOpInfo.op.setOperand(index, operand);
    }
};

void handleParams(QuantumOpInfo topEdgeOp, QuantumOpInfo bottomEdgeOp, scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter)
{
    auto topEdgeParams = topEdgeOp.op.getParams();
    auto bottomEdgeParams = bottomEdgeOp.op.getParams();

    if (topEdgeParams.size() > 0 && topEdgeParams.size() == bottomEdgeParams.size()) {
        auto cloneTopOp = topEdgeOp.op.clone();

        // check if the topEdgeParams is a tensor.extract
        for (auto [idx, param] : llvm::enumerate(topEdgeParams)) {
            if (auto tensorExtractOp = param.getDefiningOp<tensor::ExtractOp>()) {
                auto tensorExtractOpClone = tensorExtractOp.clone();
                param = tensorExtractOpClone.getResult();
                cloneTopOp.setOperand(idx, param);
                rewriter.setInsertionPoint(bottomEdgeOp.op);
                rewriter.insert(tensorExtractOpClone);
            }
            else if (auto invarOp = findInitValue(forOp, param)) {
                cloneTopOp.setOperand(idx, invarOp);
            }
            else {
                cloneTopOp.setOperand(idx, param);
            }
        }

        cloneTopOp.setQubitOperands(bottomEdgeOp.op.getInQubits());
        rewriter.setInsertionPoint(bottomEdgeOp.op);
        rewriter.insert(cloneTopOp);

        auto cloneBottomOp = bottomEdgeOp.op.clone();

        for (auto [idx, param] : llvm::enumerate(bottomEdgeParams)) {
            cloneBottomOp.setOperand(idx, param);
        }
        cloneBottomOp.setQubitOperands(cloneTopOp.getOutQubits());
        bottomEdgeOp.op.setQubitOperands(cloneBottomOp.getOutQubits());

        rewriter.setInsertionPointAfter(cloneTopOp);
        rewriter.insert(cloneBottomOp);

        // change the param of topEdgeOp to negative value
        for (auto [idx, param] : llvm::enumerate(topEdgeParams)) {
            mlir::Value negParam =
                rewriter.create<arith::NegFOp>(bottomEdgeOp.op.getLoc(), param).getResult();
            bottomEdgeOp.op.setOperand(idx, negParam);
        }
    }
}

struct LoopBoundaryForLoopRewritePattern : public mlir::OpRewritePattern<scf::ForOp> {
    using mlir::OpRewritePattern<scf::ForOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(scf::ForOp forOp,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << forOp << "\n");

        QubitOriginMap<CustomOp> topEdgeOpSet = traceTopEdgeOperations<CustomOp>(forOp);
        QubitOriginMap<CustomOp> bottomEdgeOpSet = traceBottomEdgeOperations<CustomOp>(forOp);

        auto edgeOperationSet = getVerifyEdgeOperationSet<CustomOp>(bottomEdgeOpSet, topEdgeOpSet);

        for (auto [topEdgeOp, bottomEdgeOp] : edgeOperationSet) {

            hoistTopEdgeOperation<CustomOp>(topEdgeOp, forOp, rewriter);
            handleParams(topEdgeOp, bottomEdgeOp, forOp, rewriter);
            hoistBottomEdgeOperation<CustomOp>(bottomEdgeOp, forOp, bottomEdgeOpSet, rewriter);

            return mlir::success();
        }

        return mlir::failure();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateLoopBoundaryPatterns(mlir::RewritePatternSet &patterns)
{
    patterns.add<LoopBoundaryForLoopRewritePattern>(patterns.getContext(), 1);
}

} // namespace quantum
} // namespace catalyst
