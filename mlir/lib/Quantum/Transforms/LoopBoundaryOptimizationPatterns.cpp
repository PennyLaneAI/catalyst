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

#define DEBUG_TYPE "loop-boundary"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

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

namespace {

// TODO: Add and test CRX, CRY, CRZ, ControlledPhaseShift, PhaseShift
static const mlir::StringSet<> rotationsSet = {"RX", "RY", "RZ"};
static const mlir::StringSet<> hamiltonianSet = {"Hadamard", "PauliX", "PauliY", "PauliZ",
                                                 "H",        "X",      "Y",      "Z"};
static const mlir::StringSet<> multiQubitSet = {"CNOT", "CZ", "SWAP"};

// This mode is used to determine which gates are allowed at the loop boundary.
// All: All gates are allowed
// Rotation: Only rotation gates are allowed
// Hamiltonian: Only Hamiltonian and multi-qubit gates are allowed
enum class Mode {
    All = 0,
    Rotation = 1,
    NonRotation = 2, // Always end with a Hamiltonian gate
};

// TODO: Move to a separate file
//===----------------------------------------------------------------------===//
//                        Helper functions
//===----------------------------------------------------------------------===//

// Represents the origin of a qubit, including its source register,
// position within the register, and whether it's part of a register.
struct QubitOrigin {
    mlir::Value sourceRegister;
    unsigned long position;
    bool isRegister;

    QubitOrigin() : sourceRegister(nullptr), position(0), isRegister(false) {}

    QubitOrigin(mlir::Value reg, unsigned long pos, bool isReg)
        : sourceRegister(reg), position(pos), isRegister(isReg)
    {
    }

    bool operator==(const QubitOrigin &other) const
    {
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
        : op(op), inQubits(std::move(inQubits)), isTopEdge(isTop)
    {
    }
};

// Map the operation to the list of qubit origins
template <typename OpType> using QubitOriginMap = std::map<OpType, std::vector<QubitOrigin>>;

// Checks if the given operation is a valid quantum operation based on its gate name.
template <typename OpType> bool isValidQuantumOperation(OpType &op, Mode mode)
{
    auto gateName = op.getGateName();

    switch (mode) {
    case Mode::Rotation:
        return rotationsSet.contains(gateName);
    case Mode::NonRotation:
        return hamiltonianSet.contains(gateName) || multiQubitSet.contains(gateName);
    case Mode::All:
        return hamiltonianSet.contains(gateName) || rotationsSet.contains(gateName) ||
               multiQubitSet.contains(gateName);
    }
}

// Determines if the given operation has any successors that are quantum CustomOps.
template <typename OpType> bool hasQuantumCustomSuccessor(OpType &op)
{
    return llvm::any_of(op->getUsers(), [](Operation *user) { return isa<OpType>(user); });
}

// Verifies that two sets of qubit origins are equivalent.
bool verifyQubitOrigins(const std::vector<QubitOrigin> &topOrigins,
                        const std::vector<QubitOrigin> &bottomOrigins)
{
    if (topOrigins.size() != bottomOrigins.size()) {
        return false;
    }
    return std::equal(topOrigins.begin(), topOrigins.end(), bottomOrigins.begin());
}

// Checks if the top operation has any quantum CustomOp predecessors.
template <typename OpType> bool hasQuantumCustomPredecessor(OpType &op)
{
    for (auto operand : op.getInQubits()) {
        if (auto definingOp = operand.getDefiningOp()) {
            if (isa<CustomOp>(definingOp)) {
                return true;
            }
        }
    }
    return false;
}

// Helper function to determine if a pair of operations can be optimized.
template <typename OpType>
bool isValidEdgePair(OpType &bottomOp, std::vector<QubitOrigin> &bottomQubitOrigins, OpType &topOp,
                     std::vector<QubitOrigin> &topQubitOrigins)
{
    auto bottomOpNonConst = bottomOp;
    auto topOpNonConst = topOp;
    if (bottomOpNonConst.getGateName() != topOpNonConst.getGateName() || topOp == bottomOp ||
        !verifyQubitOrigins(topQubitOrigins, bottomQubitOrigins) ||
        hasQuantumCustomSuccessor(bottomOp) || hasQuantumCustomPredecessor(topOpNonConst)) {
        return false;
    }
    return true;
}

template <typename OpType>
std::vector<std::pair<QuantumOpInfo, QuantumOpInfo>>
getVerifyEdgeOperationSet(QubitOriginMap<OpType> bottomEdgeOpSet,
                          QubitOriginMap<OpType> topEdgeOpSet)
{
    std::vector<std::pair<QuantumOpInfo, QuantumOpInfo>> edgeOperationSet;

    for (auto &[bottomOp, bottomQubitOrigins] : bottomEdgeOpSet) {
        for (auto &[topOp, topQubitOrigins] : topEdgeOpSet) {
            if (isValidEdgePair(bottomOp, bottomQubitOrigins, topOp, topQubitOrigins)) {
                QuantumOpInfo topOpInfo(topOp, topQubitOrigins, true);
                QuantumOpInfo bottomOpInfo(bottomOp, bottomQubitOrigins, false);
                edgeOperationSet.emplace_back(topOpInfo, bottomOpInfo);
            }
        }
    }
    return edgeOperationSet;
}

// Creates a quantum.extract operation.
quantum::ExtractOp createExtractOp(mlir::Value qreg, const QubitOrigin &qubit,
                                   PatternRewriter &rewriter)
{
    auto loc = qubit.sourceRegister.getLoc();
    auto idxAttr = rewriter.getI64IntegerAttr(qubit.position);
    auto type = rewriter.getType<quantum::QubitType>();
    return rewriter.create<quantum::ExtractOp>(loc, type, qreg, nullptr, idxAttr);
}

// Creates a quantum.insert operation.
quantum::InsertOp createInsertOp(mlir::Value qreg, const QubitOrigin &qubit, mlir::Value element,
                                 PatternRewriter &rewriter)
{
    assert(element && "InsertOp requires an element value!");
    auto loc = qubit.sourceRegister.getLoc();
    auto idxAttr = rewriter.getI64IntegerAttr(qubit.position);
    return rewriter.create<quantum::InsertOp>(loc, qreg.getType(), qreg, nullptr, idxAttr, element);
}

// Finds the initial value of a quantum register in the for loop.
mlir::Value findInitValue(scf::ForOp forOp, mlir::Value qReg)
{
    for (auto [arg, regionArg] : llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
        if (qReg == regionArg) {
            return arg;
        }
    }
    return nullptr;
}

// Updates the initial arguments of the for loop with a new value.
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

// Finds the result for an operation by qubit where the qubit is a region argument.
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

// Finds the root qubit by traversing backward through defining operations.
template <typename OpType> mlir::Value findRootQubit(mlir::Value qubit)
{
    mlir::Value rootQubit = qubit;
    while (auto definingOp = dyn_cast_or_null<OpType>(rootQubit.getDefiningOp())) {
        for (auto [idx, outQubit] : llvm::enumerate(definingOp.getOutQubits())) {
            if (outQubit == rootQubit) {
                rootQubit = definingOp.getInQubits()[idx];
                break;
            }
        }
    }
    return rootQubit;
}

// Determines the origin of a qubit, considering whether it's from a register.
QubitOrigin determineQubitOrigin(mlir::Value qubit)
{
    if (auto extractOp = qubit.getDefiningOp<quantum::ExtractOp>()) {
        unsigned long position = 0;
        if (extractOp.getIdxAttr().has_value()) {
            position = extractOp.getIdxAttr().value();
        }
        return QubitOrigin(extractOp.getQreg(), position, true);
    }
    return QubitOrigin(qubit, 0, false);
}

// Traces the origin of qubits for the given operation and populates the qubitOriginMap.
template <typename OpType> void traceOriginQubit(OpType &op, QubitOriginMap<OpType> &qubitOriginMap)
{
    if (qubitOriginMap.count(op))
        return;

    std::vector<QubitOrigin> qubitOrigins;
    for (auto qubit : op.getInQubits()) {
        auto rootQubit = findRootQubit<OpType>(qubit);
        qubitOrigins.push_back(determineQubitOrigin(rootQubit));
    }
    qubitOriginMap[op] = qubitOrigins;
}

// Traces quantum operations at the top edge of a loop to identify candidates for hoisting.
// Returns a map from quantum operations to the origins of their input qubits.
template <typename OpType> QubitOriginMap<OpType> traceTopEdgeOperations(scf::ForOp forOp)
{
    QubitOriginMap<OpType> qubitOriginMap;
    auto initArgs = forOp.getInitArgs();
    auto regionIterArgs = forOp.getRegionIterArgs();

    for (auto [initArg, regionArg] : llvm::zip(initArgs, regionIterArgs)) {
        mlir::Type argType = initArg.getType();

        if (isa<quantum::QuregType>(argType)) {
            for (Operation *userOp : regionArg.getUsers()) {
                if (auto extractOp = dyn_cast<quantum::ExtractOp>(userOp)) {
                    unsigned long position = 0;
                    if (extractOp.getIdxAttr().has_value()) {
                        position = extractOp.getIdxAttr().value();
                    }

                    QubitOrigin qubitOrigin(regionArg, position, true);

                    for (Operation *extractUserOp : extractOp.getResult().getUsers()) {
                        if (auto quantumOp = dyn_cast<CustomOp>(extractUserOp)) {
                            // Find the index of the extracted qubit in the
                            //  operation's input qubits.
                            auto inQubits = quantumOp.getInQubits();
                            unsigned long index = 0;
                            for (; index < inQubits.size(); ++index) {
                                if (inQubits[index] == extractOp.getResult()) {
                                    break;
                                }
                            }
                            assert(index < inQubits.size() &&
                                   "Extracted qubit not found in input qubits.");
                            if (qubitOriginMap[quantumOp].size() <= index) {
                                qubitOriginMap[quantumOp].resize(inQubits.size());
                            }
                            qubitOriginMap[quantumOp][index] = qubitOrigin;
                        }
                    }
                }
            }
        }
        // Handle single qubit arguments.
        else if (isa<quantum::QubitType>(argType)) {
            QubitOrigin qubitOrigin(regionArg, 0, false);
            for (Operation *userOp : regionArg.getUsers()) {
                if (auto quantumOp = dyn_cast<CustomOp>(userOp)) {
                    // Ensure the vector is properly sized.
                    if (qubitOriginMap.count(quantumOp) && qubitOriginMap[quantumOp].size() <= 0) {
                        qubitOriginMap[quantumOp].resize(1);
                    }
                    auto &origins = qubitOriginMap[quantumOp];
                    origins.push_back(qubitOrigin);
                }
            }
        }
    }

    return qubitOriginMap;
}

// Traces quantum operations at the bottom edge of a loop to identify candidates for hoisting.
// Returns a map from quantum operations to the origins of their input qubits.
template <typename OpType> QubitOriginMap<OpType> traceBottomEdgeOperations(scf::ForOp forOp)
{
    QubitOriginMap<OpType> qubitOriginMap;
    Operation *terminator = forOp.getBody()->getTerminator();

    for (auto operand : terminator->getOperands()) {
        Operation *definingOp = operand.getDefiningOp();
        while (InsertOp insertOp = dyn_cast_or_null<quantum::InsertOp>(definingOp)) {
            operand = insertOp.getQubit();
            auto operation = dyn_cast_or_null<OpType>(operand.getDefiningOp());
            if (!operation) {
                continue;
            }
            traceOriginQubit(operation, qubitOriginMap);
            definingOp = insertOp.getInQreg().getDefiningOp();
        }

        auto operation = dyn_cast_or_null<OpType>(operand.getDefiningOp());

        if (!operation) {
            continue;
        }
        traceOriginQubit(operation, qubitOriginMap);
    }
    return qubitOriginMap;
}

// Hoists a quantum operation from the top edge of a loop to the top of the loop.
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
            auto extractOp = createExtractOp(initValue, qubit, rewriter);
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
            auto tensor = extractOp.getTensor();
            if (auto initParam = findInitValue(forOp, tensor)) {
                extractOpClone->setOperands(initParam);
            }
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
            auto insertOp = createInsertOp(reg, qubit, topOpInfo.op.getOutQubits()[idx], rewriter);
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

// Hoists a quantum operation from the bottom edge of a loop to the bottom of the loop.
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
        auto regionIterArgs = forOp.getRegionIterArgs();
        if (std::find(regionIterArgs.begin(), regionIterArgs.end(), param) !=
            regionIterArgs.end()) {
            bottomOpInfo.op.setOperand(idx, param);
            continue;
        }
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
            auto insertOp = createInsertOp(reg, qubit, outQubit, rewriter);
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
            auto extractOp = createExtractOp(reg, qubit, rewriter);
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

// This function aims to set the operands of the cloneOp with paramOp's params.
void setParamOperation(mlir::Operation &cloneOp, QuantumOpInfo paramOp, QuantumOpInfo bottomEdgeOp,
                       scf::ForOp forOp, mlir::PatternRewriter &rewriter)
{
    for (auto [idx, param] : llvm::enumerate(paramOp.op.getParams())) {
        auto regionIterArgs = forOp.getRegionIterArgs();
        // If param is in forOp, then it is a loop invariant
        if (std::find(regionIterArgs.begin(), regionIterArgs.end(), param) !=
            regionIterArgs.end()) {
            cloneOp.setOperand(idx, param);
            continue;
        }
        if (auto tensorExtractOp = param.getDefiningOp<tensor::ExtractOp>()) {
            auto tensorExtractOpClone = tensorExtractOp.clone();
            param = tensorExtractOpClone.getResult();
            cloneOp.setOperand(idx, param);
            rewriter.setInsertionPoint(bottomEdgeOp.op);
            rewriter.insert(tensorExtractOpClone);
        }
        else if (auto invarOp = findInitValue(forOp, param)) {
            cloneOp.setOperand(idx, invarOp);
        }
        else {
            cloneOp.setOperand(idx, param);
        }
    }
}

// Handles parameter adjustments when moving operations across loop boundaries.
void handleParams(QuantumOpInfo topEdgeOp, QuantumOpInfo bottomEdgeOp, scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter)
{
    auto topEdgeParams = topEdgeOp.op.getParams();
    auto bottomEdgeParams = bottomEdgeOp.op.getParams();

    if (topEdgeParams.size() > 0 && topEdgeParams.size() == bottomEdgeParams.size()) {
        // Create the clone of the top edge operation
        auto cloneTopOp = topEdgeOp.op.clone();
        setParamOperation(*cloneTopOp, topEdgeOp, bottomEdgeOp, forOp, rewriter);
        cloneTopOp.setQubitOperands(bottomEdgeOp.op.getInQubits());
        rewriter.setInsertionPoint(bottomEdgeOp.op);
        rewriter.insert(cloneTopOp);

        // Create the clone of the bottom edge operation
        auto cloneBottomOp = bottomEdgeOp.op.clone();
        setParamOperation(*cloneBottomOp, bottomEdgeOp, bottomEdgeOp, forOp, rewriter);
        cloneBottomOp.setQubitOperands(cloneTopOp.getOutQubits());
        bottomEdgeOp.op.setQubitOperands(cloneBottomOp.getOutQubits());

        rewriter.setInsertionPoint(bottomEdgeOp.op);
        rewriter.insert(cloneBottomOp);

        // Update the param of topEdgeOp to negative value
        for (auto [idx, param] : llvm::enumerate(topEdgeParams)) {
            mlir::Value negParam =
                rewriter.create<arith::NegFOp>(cloneTopOp.getLoc(), param).getResult();
            rewriter.moveOpBefore(negParam.getDefiningOp(), cloneTopOp);
            bottomEdgeOp.op.setOperand(idx, negParam);
        }
    }
}

//===----------------------------------------------------------------------===//
//                   Loop Boundary Optimization Patterns
//===----------------------------------------------------------------------===//

struct LoopBoundaryForLoopRewritePattern : public mlir::OpRewritePattern<scf::ForOp> {
    // using mlir::OpRewritePattern<scf::ForOp>::OpRewritePattern;

    Mode mode;

    LoopBoundaryForLoopRewritePattern(mlir::MLIRContext *context, Mode loopBoundaryMode,
                                      mlir::PatternBenefit benefit = 1)
        : mlir::OpRewritePattern<scf::ForOp>(context, benefit), mode(loopBoundaryMode)
    {
    }

    mlir::LogicalResult matchAndRewrite(scf::ForOp forOp,
                                        mlir::PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(dbgs() << "Simplifying the following operation:\n" << forOp << "\n");

        QubitOriginMap<CustomOp> topEdgeOpSet = traceTopEdgeOperations<CustomOp>(forOp);
        QubitOriginMap<CustomOp> bottomEdgeOpSet = traceBottomEdgeOperations<CustomOp>(forOp);

        auto edgeOperationSet = getVerifyEdgeOperationSet<CustomOp>(bottomEdgeOpSet, topEdgeOpSet);

        for (auto [topEdgeOp, bottomEdgeOp] : edgeOperationSet) {
            if (isValidQuantumOperation<CustomOp>(topEdgeOp.op, mode)) {
                hoistTopEdgeOperation<CustomOp>(topEdgeOp, forOp, rewriter);
                handleParams(topEdgeOp, bottomEdgeOp, forOp, rewriter);
                hoistBottomEdgeOperation<CustomOp>(bottomEdgeOp, forOp, bottomEdgeOpSet, rewriter);
                return mlir::success();
            }
        }

        return mlir::failure();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateLoopBoundaryPatterns(mlir::RewritePatternSet &patterns, unsigned int mode)
{
    if (mode > static_cast<unsigned int>(Mode::NonRotation)) {
        llvm::errs() << "Invalid mode value: " << mode << ". Defaulting to Mode::All.\n";
        mode = static_cast<unsigned int>(Mode::All);
    }
    Mode loopBoundaryMode = static_cast<Mode>(mode);
    patterns.add<LoopBoundaryForLoopRewritePattern>(patterns.getContext(), loopBoundaryMode, 1);
}

} // namespace quantum
} // namespace catalyst
