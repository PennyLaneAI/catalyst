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
#include "mlir/Support/LogicalResult.h"

#include "Quantum/IR/QuantumOps.h"
#include "Quantum/Transforms/Patterns.h"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::quantum;

namespace {

// TODO: Add and test CRX, CRY, CRZ, ControlledPhaseShift, PhaseShift
static const StringSet<> rotationsSet = {"RX", "RY", "RZ"};
static const StringSet<> hermitianSet = {"Hadamard", "PauliX", "PauliY", "PauliZ",
                                         "H",        "X",      "Y",      "Z"};
static const StringSet<> multiQubitSet = {"CNOT", "CZ", "SWAP"};

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
    Value qubitOrRegister;
    unsigned long position;
    bool isRegister;

    QubitOrigin() : qubitOrRegister(nullptr), position(0), isRegister(false) {}

    QubitOrigin(Value val, unsigned long pos, bool isReg)
        : qubitOrRegister(val), position(pos), isRegister(isReg)
    {
    }

    bool operator==(const QubitOrigin &other) const
    {
        if (this->qubitOrRegister == nullptr || other.qubitOrRegister == nullptr) {
            return false;
        }
        return qubitOrRegister == other.qubitOrRegister && position == other.position;
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
using QubitOriginMap = std::map<CustomOp, std::vector<QubitOrigin>>;

// Checks if the given operation is a valid quantum operation based on its gate name.
bool isValidQuantumOperation(CustomOp &op, Mode mode)
{
    auto gateName = op.getGateName();

    switch (mode) {
    case Mode::Rotation:
        return rotationsSet.contains(gateName);
    case Mode::NonRotation:
        return hermitianSet.contains(gateName) || multiQubitSet.contains(gateName);
    case Mode::All:
        return hermitianSet.contains(gateName) || rotationsSet.contains(gateName) ||
               multiQubitSet.contains(gateName);
    }

    assert(false && "Invalid Enum value for `Mode`");
}

// Determines if the given operation has any successors that are quantum CustomOps.
bool hasQuantumCustomSuccessor(const CustomOp &op)
{
    return llvm::any_of(op->getUsers(), [](Operation *user) { return isa<CustomOp>(user); });
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
bool hasQuantumCustomPredecessor(CustomOp &op)
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

// Return true if bottomOp is the immediate successor of topOp.
// In this case, we don't want to perform loop boundary commutation,
// as it would prevent a fix-point from being reached by the rewrite driver.
bool isSourceAndSink(CustomOp topOp, CustomOp bottomOp)
{
    for (auto [outQubit, inQubit] : llvm::zip(topOp.getOutQubits(), bottomOp.getInQubits())) {
        if (outQubit != inQubit) {
            return false;
        }
    }
    return true;
}

// Helper function to determine if a pair of operations can be optimized.
bool isValidEdgePair(const CustomOp &bottomOp, std::vector<QubitOrigin> &bottomQubitOrigins,
                     const CustomOp &topOp, std::vector<QubitOrigin> &topQubitOrigins)
{
    auto bottomOpNonConst = bottomOp;
    auto topOpNonConst = topOp;

    if (bottomOpNonConst.getGateName() != topOpNonConst.getGateName() || topOp == bottomOp ||
        !verifyQubitOrigins(topQubitOrigins, bottomQubitOrigins) ||
        isSourceAndSink(topOpNonConst, bottomOpNonConst) || hasQuantumCustomSuccessor(bottomOp) ||
        hasQuantumCustomPredecessor(topOpNonConst)) {
        return false;
    }
    return true;
}

std::vector<std::pair<QuantumOpInfo, QuantumOpInfo>>
getVerifyEdgeOperationSet(QubitOriginMap bottomEdgeOpSet, QubitOriginMap topEdgeOpSet)
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
quantum::ExtractOp createExtractOp(Value qreg, const QubitOrigin &qubit, PatternRewriter &rewriter)
{
    auto loc = qubit.qubitOrRegister.getLoc();
    auto idxAttr = rewriter.getI64IntegerAttr(qubit.position);
    auto type = rewriter.getType<quantum::QubitType>();
    return rewriter.create<quantum::ExtractOp>(loc, type, qreg, nullptr, idxAttr);
}

// Creates a quantum.insert operation.
quantum::InsertOp createInsertOp(Value qreg, const QubitOrigin &qubit, Value element,
                                 PatternRewriter &rewriter)
{
    assert(element && "InsertOp requires an element value!");
    auto loc = qubit.qubitOrRegister.getLoc();
    auto idxAttr = rewriter.getI64IntegerAttr(qubit.position);
    return rewriter.create<quantum::InsertOp>(loc, qreg.getType(), qreg, nullptr, idxAttr, element);
}

// Finds the initial value of a quantum register in the for loop.
Value findInitValue(scf::ForOp forOp, Value qReg)
{
    for (auto [arg, regionArg] : llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
        if (qReg == regionArg) {
            return arg;
        }
    }
    return nullptr;
}

// Updates the initial arguments of the for loop with a new value.
void updateForLoopInitArg(scf::ForOp forOp, Value sourceRegister, Value newValue,
                          PatternRewriter &rewriter)
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
Value findResultForOpByQubit(Value qubit, scf::ForOp forOp)
{
    Value result;
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
Value findRootQubit(Value qubit)
{
    if (auto definingOp = dyn_cast_or_null<CustomOp>(qubit.getDefiningOp())) {
        for (auto [idx, outQubit] : llvm::enumerate(definingOp.getOutQubits())) {
            if (outQubit == qubit) {
                return findRootQubit(definingOp.getInQubits()[idx]);
            }
        }
    }
    return qubit;
}

// Determines the origin of a qubit, considering whether it's from a register.
QubitOrigin determineQubitOrigin(Value qubit)
{
    if (auto extractOp = qubit.getDefiningOp<quantum::ExtractOp>()) {
        // Dynamic register indices are unknown and should not match against other qubit origins.
        if (!extractOp.getIdxAttr().has_value()) {
            return QubitOrigin();
        }

        unsigned long position = extractOp.getIdxAttr().value();
        auto regType = extractOp.getQreg();
        while (InsertOp insertOp = regType.getDefiningOp<quantum::InsertOp>()) {
            regType = insertOp.getInQreg();
        }

        return QubitOrigin(regType, position, true);
    }

    return QubitOrigin(qubit, 0, false);
}

// Traces the origin of qubits for the given operation and populates the qubitOriginMap.
void traceOriginQubit(CustomOp &op, QubitOriginMap &qubitOriginMap, Mode mode)
{
    if (qubitOriginMap.count(op))
        return;

    if (!isValidQuantumOperation(op, mode)) {
        return;
    }

    std::vector<QubitOrigin> qubitOrigins;
    for (auto qubit : op.getInQubits()) {
        auto rootQubit = findRootQubit(qubit);
        qubitOrigins.push_back(determineQubitOrigin(rootQubit));
    }
    qubitOriginMap[op] = qubitOrigins;
}

// Traces quantum operations at the top edge of a loop to identify candidates for hoisting.
// Returns a map from quantum operations to the origins of their input qubits.
QubitOriginMap traceTopEdgeOperations(scf::ForOp forOp, Mode mode)
{
    QubitOriginMap qubitOriginMap;
    auto regionIterArgs = forOp.getRegionIterArgs();

    for (auto regionArg : regionIterArgs) {
        Type argType = regionArg.getType();

        if (isa<quantum::QuregType>(argType)) {
            for (Operation *userOp : regionArg.getUsers()) {
                if (auto extractOp = dyn_cast<quantum::ExtractOp>(userOp)) {
                    if (!extractOp.getIdxAttr().has_value())
                        continue;

                    unsigned long position = extractOp.getIdxAttr().value();
                    QubitOrigin qubitOrigin(regionArg, position, true);

                    for (Operation *extractUserOp : extractOp.getResult().getUsers()) {
                        if (auto quantumOp = dyn_cast<CustomOp>(extractUserOp)) {
                            if (!isValidQuantumOperation(quantumOp, mode))
                                continue;

                            // Find the index of the extracted qubit in the
                            //  operation's input qubits.
                            auto inQubits = quantumOp.getInQubits();
                            unsigned long index = 0;
                            for (; index < inQubits.size(); ++index) {
                                if (inQubits[index] == extractOp.getResult())
                                    break;
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
                    if (!isValidQuantumOperation(quantumOp, mode))
                        continue;

                    qubitOriginMap[quantumOp].push_back(qubitOrigin);
                }
            }
        }
    }
    return qubitOriginMap;
}

// Traces quantum operations at the bottom edge of a loop to identify candidates for hoisting.
// Returns a map from quantum operations to the origins of their input qubits.
QubitOriginMap traceBottomEdgeOperations(scf::ForOp forOp, Mode mode)
{
    QubitOriginMap qubitOriginMap;
    Operation *terminator = forOp.getBody()->getTerminator();

    for (auto operand : terminator->getOperands()) {
        Operation *definingOp = operand.getDefiningOp();
        while (InsertOp insertOp = dyn_cast_or_null<quantum::InsertOp>(definingOp)) {
            auto operation = dyn_cast_or_null<CustomOp>(insertOp.getQubit().getDefiningOp());
            if (!operation) {
                break;
            }
            traceOriginQubit(operation, qubitOriginMap, mode);
            operand = insertOp.getQubit();
            definingOp = insertOp.getInQreg().getDefiningOp();
        }

        auto operation = dyn_cast_or_null<CustomOp>(operand.getDefiningOp());

        if (!operation) {
            continue;
        }
        traceOriginQubit(operation, qubitOriginMap, mode);
    }
    return qubitOriginMap;
}

// Hoists a quantum operation from the top edge of a loop to the top of the loop.
void hoistTopEdgeOperation(QuantumOpInfo topOpInfo, scf::ForOp forOp, PatternRewriter &rewriter)
{
    rewriter.moveOpBefore(topOpInfo.op, forOp);
    topOpInfo.op.getOutQubits().replaceAllUsesWith(topOpInfo.op.getInQubits()); // config successor

    // hoist the extract operations
    std::vector<Value> operandOps;
    for (auto qubit : topOpInfo.inQubits) {
        Value initValue = findInitValue(forOp, qubit.qubitOrRegister);
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
            Value reg = findInitValue(forOp, qubit.qubitOrRegister);
            auto insertOp = createInsertOp(reg, qubit, topOpInfo.op.getOutQubits()[idx], rewriter);
            // Find the matching init arg index and update it
            updateForLoopInitArg(forOp, qubit.qubitOrRegister, insertOp.getResult(), rewriter);
            rewriter.moveOpBefore(insertOp, forOp);
        }
        else {
            // if it is not a register, it is a qubit. so don't need to create vector operation
            // just need to find the matching init arg index and update it
            updateForLoopInitArg(forOp, qubit.qubitOrRegister, topOpInfo.op.getOutQubits()[idx],
                                 rewriter);
        }
    }
};

// Hoists a quantum operation from the bottom edge of a loop to the bottom of the loop.
void hoistBottomEdgeOperation(QuantumOpInfo bottomOpInfo, scf::ForOp forOp,
                              QubitOriginMap bottomEdgeOpSet, PatternRewriter &rewriter)
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
        if (auto op = dyn_cast_or_null<Operation *>(param.getDefiningOp())) {
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
        Value reg;
        reg = findResultForOpByQubit(qubit.qubitOrRegister, forOp);

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
    std::vector<Value> operandOps;
    for (auto qubit : bottomOpInfo.inQubits) {
        // inint value should be forOp result
        Value reg = findResultForOpByQubit(qubit.qubitOrRegister, forOp);

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
void setParamOperation(Operation &cloneOp, QuantumOpInfo paramOp, QuantumOpInfo bottomEdgeOp,
                       scf::ForOp forOp, PatternRewriter &rewriter)
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
                  PatternRewriter &rewriter)
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
            Value negParam = rewriter.create<arith::NegFOp>(cloneTopOp.getLoc(), param).getResult();
            rewriter.moveOpBefore(negParam.getDefiningOp(), cloneTopOp);
            bottomEdgeOp.op.setOperand(idx, negParam);
        }
    }
}

//===----------------------------------------------------------------------===//
//                   Loop Boundary Optimization Patterns
//===----------------------------------------------------------------------===//

struct LoopBoundaryForLoopRewritePattern : public OpRewritePattern<scf::ForOp> {
    Mode mode;

    LoopBoundaryForLoopRewritePattern(MLIRContext *context, Mode loopBoundaryMode,
                                      PatternBenefit benefit = 1)
        : OpRewritePattern<scf::ForOp>(context, benefit), mode(loopBoundaryMode)
    {
    }

    LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override
    {
        LLVM_DEBUG(llvm::dbgs() << "Simplifying the following operation:\n" << forOp << "\n");

        QubitOriginMap topEdgeOpSet = traceTopEdgeOperations(forOp, mode);
        QubitOriginMap bottomEdgeOpSet = traceBottomEdgeOperations(forOp, mode);

        auto edgeOperationSet = getVerifyEdgeOperationSet(bottomEdgeOpSet, topEdgeOpSet);

        if (edgeOperationSet.empty()) {
            return failure();
        }

        for (auto [topEdgeOp, bottomEdgeOp] : edgeOperationSet) {
            hoistTopEdgeOperation(topEdgeOp, forOp, rewriter);
            handleParams(topEdgeOp, bottomEdgeOp, forOp, rewriter);
            hoistBottomEdgeOperation(bottomEdgeOp, forOp, bottomEdgeOpSet, rewriter);
        }

        return success();
    }
};

} // namespace

namespace catalyst {
namespace quantum {

void populateLoopBoundaryPatterns(RewritePatternSet &patterns, unsigned int mode)
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
