// Copyright 2026 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define DEBUG_TYPE "value-semantics-conversion"

#include "value_semantics_conversion.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"

#include "MBQC/IR/MBQCOps.h"
#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/IR/QRefTypes.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"
#include "Quantum/IR/QuantumTypes.h"

using namespace mlir;
using namespace catalyst;

// In this file, variable names like "vQubit" stand for "qubits in value semantics",
// and variable names like "rQubit" stand for "qubits in reference semantics".

namespace {

// A struct to store the register and the index of rQubits from a qref.get operation.
// This struct is intended to be the keys in `llvm::DenseMap`s.
struct rQubitGetOpInfo {
    Value reg;
    int64_t idxAttr;
    Value idx;

    rQubitGetOpInfo(Value _reg, Value _idx) : reg(_reg), idxAttr(-1), idx(_idx) {}

    rQubitGetOpInfo(Value _reg, int64_t _idxAttr) : reg(_reg), idxAttr(_idxAttr), idx(nullptr) {}

    bool operator==(const rQubitGetOpInfo &other) const
    {
        return reg == other.reg && idxAttr == other.idxAttr && idx == other.idx;
    }
};

std::optional<rQubitGetOpInfo> getGetOpInfo(Value rQubit)
{
    bool isGetOp = rQubit.getDefiningOp() && isa<qref::GetOp>(rQubit.getDefiningOp());
    if (!isGetOp) {
        return std::nullopt;
    }

    auto getOp = cast<qref::GetOp>(rQubit.getDefiningOp());
    Value reg = getOp.getQreg();
    if (getOp.getIdxAttr().has_value()) {
        return rQubitGetOpInfo(reg, getOp.getIdxAttr().value());
    }
    else {
        return rQubitGetOpInfo(reg, getOp.getIdx());
    }
}

/**
 * @brief Given a non-root rQubit Value, return the rQreg Value that it belongs to.
 * The non-root rQubit Value must be the result of a qref.get op.
 *
 * @param rQubit
 * @return Value
 */
Value getRSourceRegisterValue(Value rQubit)
{
    assert(isa<qref::QubitType>(rQubit.getType()) &&
           "Can only query qref.bit types for source qref.reg values");
    auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
    assert(getOp && "Expected a non-root rQubit coming from a qref.get op");
    return getOp.getQreg();
}

/**
 * @brief Given a qref gate operation, compute the result segment sizes for the corresponding value
 * semantics gate operation.
 *
 * The reference semantics gates do not produce results.
 * Therefore, we need to manually set the result segment sizes for the corresponding value semantics
 * gate,
 *
 * @param builder
 * @param rGateOp
 * @return DenseI32ArrayAttr
 */
DenseI32ArrayAttr getResultSegmentSizes(IRRewriter &builder, qref::QuantumGate rGateOp)
{
    int32_t non_ctrl_len = rGateOp.getNonCtrlQubitOperands().size();
    int32_t ctrl_len = rGateOp.getCtrlQubitOperands().size();
    return builder.getDenseI32ArrayAttr({non_ctrl_len, ctrl_len});
}

/**
 * @brief Determine whether a func op is a qref subroutine that needs semantics conversion.
 *
 * @param f
 */
bool isQrefSubroutine(func::FuncOp f)
{
    // quantum.node is not a subroutine
    if (f->hasAttrOfType<UnitAttr>("quantum.node")) {
        return false;
    }

    // If has a qref argument, definitely is a qref subroutine
    if (llvm::any_of(f.getArgumentTypes(), llvm::IsaPred<qref::QubitType, qref::QuregType>)) {
        return true;
    }

    // If we don't know from the args, must look at the body
    if (f.isDeclaration()) {
        return false;
    }

    WalkResult walkResult = f.walk([](Operation *op) {
        if (isa<qref::QRefDialect>(op->getDialect())) {
            return WalkResult::interrupt();
        }
        if (func::CallOp callOp = dyn_cast<func::CallOp>(op)) {
            auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                callOp, callOp.getCalleeAttr());
            assert(funcOp && "calling a non-existent subroutine");
            if (isQrefSubroutine(funcOp)) {
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return walkResult.wasInterrupted();
}

/**
 * @brief Erase all remaining qref.alloc, qref.get, qref.mbqc.graph_state_prep operations in a
 * function
 *
 * During the conversion, these operations cannot be deleted immediately because they might be used
 * by other qref operations after their own conversion is done.
 * This utility erases all of them, and must only be called at the end of a function's conversion.
 */
void eraseAllRemainingAnchorRValues(func::FuncOp f)
{
    f.walk([&](qref::GetOp getOp) {
        assert(getOp.use_empty() &&
               "qref.bit Values must have no uses after the semantic conversion");
        getOp->erase();
    });
    f.walk([&](qref::AllocOp allocOp) {
        assert(allocOp.use_empty() &&
               "qref.reg Values must have no uses after the semantic conversion");
        allocOp->erase();
    });
    f.walk([&](qref::GraphStatePrepOp graphStatePrepOp) {
        assert(graphStatePrepOp.use_empty() &&
               "qref.reg Values must have no uses after the semantic conversion");
        graphStatePrepOp->erase();
    });
}

/**
 * @brief This struct tracks the current vQreg and vQubit Values for the root rQreg and rQubit
 * Values in a given region. There should be exactly one tracker instance per region.
 *
 * In a region, the root rQreg and rQubit values can come from one of two places:
 * - An allocation, i.e. a qref.alloc or qref.alloc_qb operation
 * - An argument, e.g. arguments to a subroutine or control flow region
 *
 * These Values are considered "root" because they are guaranteed to be uniquely defined: different
 * allocations and arguments are semantically distinct. By contrast, rQubit Values returned from
 * qref.get operations are not considered root values, because multiple qref.get operations can
 * alias each other.
 *
 * Importantly, vQubit Values extracted from a vQreg are not tracked. Instead, each usage of the
 * corresponding rQubit Value (from a qref.get operation) will extract the vQubit Values before the
 * use, and insert the vQubit Values after the use.
 *
 * Note that the above distinction of root and non-root values is not universal. Some rQubits from
 * qref.get ops can also be considered root. For example, since scf.if operations do not take in
 * actual operands, the "argument rQubits" (which are supposed to be root from the above guidelines,
 * since it is a scope argument in spirit) must be taken in via closure, regardless of whether these
 * rQubits came from qref.get or allocation/arguments in the parent scope.
 *
 * The actual source of truth for the distinction is, if an rValue is recorded in the Tracker's
 * maps, then it is considered root.
 *
 * For root rQubits coming from qref.get operations, the map key is their `rQubitGetOpInfo` objects,
 * as opposed to their SSA Value themselves. This ensures the multiple aliasing qref.get operations
 * get mapped to the same current vQubit SSA Value.
 *
 * It is up to the callers of this Tracker class to decide which rValues are root in their specific
 * circumstances.
 */
struct QubitValueTracker {
  public:
    QubitValueTracker() = default;

    /**
     * @brief Return the current vQreg Value corresponding to a root rQreg Value.
     *
     * @param rQreg
     * @return const Value
     */
    const Value getCurrentVQreg(Value rQreg)
    {
        assert(isa<qref::QuregType>(rQreg.getType()) && "Expected qref.reg type");

        Value vQreg = this->qreg_map.at(rQreg);
        assert(isa<quantum::QuregType>(vQreg.getType()) && "Expected quantum.reg type");
        return vQreg;
    }

    /**
     * @brief Set the current vQreg Value corresponding to a root rQreg Value.
     *
     * @param rQreg
     * @param vQreg
     */
    void setCurrentVQreg(Value rQreg, Value vQreg)
    {
        assert(isa<qref::QuregType>(rQreg.getType()) && "Expected qref.reg type");
        assert(isa<quantum::QuregType>(vQreg.getType()) && "Expected quantum.reg type");
        this->qreg_map[rQreg] = vQreg;
    }

    /**
     * @brief Return the current vQubit Value corresponding to a root rQubit Value.
     *
     * @param rQubit
     * @return const Value
     */
    const Value getCurrentVQubit(Value rQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        assert(this->isRootRQubit(rQubit) && "The provided qref.bit value is not root");

        std::optional<rQubitGetOpInfo> getOpInfo = getGetOpInfo(rQubit);

        Value vQubit;
        if (!getOpInfo.has_value()) {
            vQubit = this->qubit_map.at(rQubit);
        }
        else {
            vQubit = this->getOp_qubit_map.at(getOpInfo.value());
        }

        assert(isa<quantum::QubitType>(vQubit.getType()) && "Expected quantum.bit type");
        return vQubit;
    }

    /**
     * @brief Set the current vQubit Value corresponding to a root rQubit Value.
     *
     * @param rQubit
     * @param vQubit
     */
    void setCurrentVQubit(Value rQubit, Value vQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        assert(isa<quantum::QubitType>(vQubit.getType()) && "Expected quantum.bit type");

        std::optional<rQubitGetOpInfo> getOpInfo = getGetOpInfo(rQubit);

        if (!getOpInfo.has_value()) {
            this->qubit_map[rQubit] = vQubit;
        }
        else {
            this->getOp_qubit_map[getOpInfo.value()] = vQubit;
        }
    }

    bool isRootRQubit(Value rQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");

        std::optional<rQubitGetOpInfo> getOpInfo = getGetOpInfo(rQubit);

        if (!getOpInfo) {
            return this->qubit_map.contains(rQubit);
        }
        else {
            return this->getOp_qubit_map.contains(getOpInfo.value());
        }
    }

  private:
    // The map is from root qref.qreg values of a region to the current quantum.qreg values.
    llvm::DenseMap<Value, Value> qreg_map;

    // The map is from root qref.bit values of a region to the current quantum.bit values.
    llvm::DenseMap<Value, Value> qubit_map;

    // This map records the root qubits coming from qref.get ops.
    // The key is a pair of the register and the index on the get op, so that we can handle aliasing
    llvm::DenseMap<rQubitGetOpInfo, Value> getOp_qubit_map;
}; // struct QubitValueTracker

/**
 * @brief This struct is responsible for extracting and inserting vQubits before and after uses.
 *
 * @details This class must be the sole source of the creation of quantum.extract and insert
 * operations.
 *
 * We follow a very simple strategy in this semantics conversion, which we now describe.
 *
 * Whenever a qref operation (gate, observables, or any region containing those) uses some rQubit
 * and rQreg Values, these Values are either root or non-root.
 *
 * Root Values are tracked in the tracker maps directly and do not need to be extracted, as they
 * have no source registers. Note that all rQreg Values are root.
 *
 * Non-root rQubit Values are all considered transient. That is, their corresponding vQubit Values
 * are extracted immediately before the uses (from their source rQreg Value's current vQreg Value),
 * and inserted back immediately after the use. They are not tracked globally across uses, unlike
 * the root Values.
 *
 * If the given user op consumes the extracted vQubits but does not return vQubit results (e.g.
 * observable ops), the extracted vQubits are inserted.
 *
 * The order of the extract and insert operations created are in the order that the non-root
 * rQubit Values appear in the user operation.
 *
 * When an operation (rOp) using non-root rQubit Values is interfacing with this struct, it must
 * guarantee the following are true on the new operation (vOp) it creates in the value semantics
 * dialect:
 * 1. If the rOp returns any results, the vOp must return results of the same types at the same
 * result indices. This is because even though a rOp cannot return any qubit or qreg Values, it
 * can still return classical Values (e.g. from regions) or observables (e.g. from namedobs op).
 * 2. Any vQubit and vQreg Values returned from vOp must appear at the end of the vOp's list of
 * results, after any existing result Values from the previous point, without being interleaved
 * with any other results.
 * 3. The order in which these new vQubit and vQreg results appear must match the order in which
 * they appear in the rOp's operands.
 *
 * For example, denoting the 3 kinds (qreg, root qubit, non-root qubit) by A, B, C, and classical
 * Values by regular names, the valid vOp for the following rOp
 *   %0, %1 = qref.some_op %arg0, %A1, %arg1, %B1, %B2, %C1, %A2, %arg2, %C2
 * is
 *   %0, %1, %A1_out, %B1_out, %B2_out, %C1_out, %A2_out, %C2_out = quantum.some_op ....
 *
 * This struct offers methods to query the expected indices that follow the above convention.
 * The "getOperandIndices" methods return the operand indices of requested kind on the rOp.
 * The "getResultIndices" methods return the expected result indices of requested kind on the vOp.
 * In the above example, the methods would return:
 * getQRegOperandIndices(): [1, 6]
 * getRootQubitOperandIndices(): [3, 4]
 * getNonRootQubitOperandIndices(): [5, 8]
 * getQRegResultIndices(): [2, 6]
 * getRootQubitResultIndices(): [3, 4]
 * getNonRootQubitResultIndices(): [5, 7]
 *
 * This struct can also be used with the operands being passed in explicitly. This is useful when
 * treating regions (e.g. control flow) as "big gates", since they don't have the rValues needed
 * from above properly captured as true operands.
 *
 * @param rOp
 * @param tracker
 * @param builder
 */
struct TransientQubitExtractor {
  public:
    TransientQubitExtractor(QubitValueTracker &_tracker, IRRewriter &_builder, Operation *_rOp,
                            std::optional<ArrayRef<Value>> explicit_operands = std::nullopt)
        : rOp(_rOp), tracker(_tracker), builder(_builder)
    {
        OpBuilder::InsertionGuard guard(this->builder);
        this->builder.setInsertionPoint(this->rOp);

        this->analyzeROpQuantumOperandPatterns(explicit_operands);

        ValueRange rOpOperands = this->rOp->getOperands();
        ValueRange operands =
            explicit_operands.has_value() ? explicit_operands.value() : rOpOperands;
        for (auto [i, operand] : llvm::enumerate(operands)) {
            if (!isa<qref::QubitType>(operand.getType()) || this->tracker.isRootRQubit(operand)) {
                continue;
            }

            this->sourceRQregs.push_back(getRSourceRegisterValue(operand));
            this->extractOps.push_back(this->createExtractOp(operand));
        }
    }

    ~TransientQubitExtractor()
    {
        if (this->getNonRootQubitResultIndices().size() == 0) {
            // Nothing to insert
            return;
        }

        OpBuilder::InsertionGuard guard(this->builder);
        this->builder.setInsertionPointAfter(this->rOp);

        assert(this->vOp && "The converted operation in value semantics was not provided");

        // 1. Fetch the vQubit results from the newly created value semantics user op
        // Only need to insert non-root rQubits.
        //
        // e.g. if the new vOp is
        // %classical, %vQreg, %vQubit_non_root0, %vQubit_root, %vQubit_non_root1
        //  = quantum.custom .... %inQreg, %inQubit_non_root0, %inQubit_root, %inQubit_non_root1
        //
        // we need to insert result Values at indices 2 and 4
        std::vector<unsigned> nonRootQubitResultIndices = this->getNonRootQubitResultIndices();

        // 2. Create the insert ops
        for (auto triplet :
             llvm::zip_equal(this->extractOps, this->sourceRQregs, nonRootQubitResultIndices)) {
            quantum::ExtractOp extractOp = std::get<0>(triplet);
            Value sourceRQreg = std::get<1>(triplet);
            unsigned resultIdx = std::get<2>(triplet);

            Value qubitToInsert;
            if (resultIdx >= vOp->getNumResults()) {
                qubitToInsert = extractOp.getQubit();
            }
            else {
                qubitToInsert = vOp->getResult(resultIdx);
            }

            auto insertOp = quantum::InsertOp::create(
                this->builder, extractOp->getLoc(), quantum::QuregType::get(extractOp.getContext()),
                this->tracker.getCurrentVQreg(sourceRQreg), extractOp.getIdx(),
                extractOp.getIdxAttrAttr(), qubitToInsert);
            this->tracker.setCurrentVQreg(sourceRQreg, insertOp.getOutQreg());
            this->builder.setInsertionPointAfter(insertOp);
        }
    }

    /**
     * @brief Get the extracted vQubit Values.
     *
     * @return SmallVector<Value>
     */
    SmallVector<Value> getExtractedVQubits()
    {
        SmallVector<Value> result;
        for (quantum::ExtractOp &extractOp : this->extractOps) {
            result.push_back(extractOp.getQubit());
        }
        return result;
    }

    const std::vector<unsigned> &getQRegOperandIndices() { return this->rQregOperandIndices; }

    const std::vector<unsigned> &getRootQubitOperandIndices()
    {
        return this->rootRQubitOperandIndices;
    }

    const std::vector<unsigned> &getNonRootQubitOperandIndices()
    {
        return this->nonRootRQubitOperandIndices;
    }

    const std::vector<unsigned> &getQRegResultIndices() { return this->vQregResultIndices; }

    const std::vector<unsigned> &getRootQubitResultIndices()
    {
        return this->rootVQubitResultIndices;
    }

    const std::vector<unsigned> &getNonRootQubitResultIndices()
    {
        return this->nonRootVQubitResultIndices;
    }

    void setVOp(Operation *vOp) { this->vOp = vOp; }

  private:
    Operation *rOp;
    Operation *vOp = nullptr;
    QubitValueTracker &tracker;
    IRRewriter &builder;

    std::vector<unsigned> nonRootRQubitOperandIndices;
    std::vector<unsigned> rootRQubitOperandIndices;
    std::vector<unsigned> rQregOperandIndices;
    std::vector<unsigned> nonRootVQubitResultIndices;
    std::vector<unsigned> rootVQubitResultIndices;
    std::vector<unsigned> vQregResultIndices;

    SmallVector<Value> sourceRQregs;
    SmallVector<quantum::ExtractOp> extractOps;

    void analyzeROpQuantumOperandPatterns(std::optional<ArrayRef<Value>> explicit_operands)
    {
        unsigned resIdx = 0;
        unsigned existingNumResults = this->rOp->getNumResults();

        SmallVector<Value> operands;
        if (!explicit_operands.has_value()) {
            operands.append(this->rOp->getOperands().begin(), this->rOp->getOperands().end());
        }
        else {
            operands.append(explicit_operands.value().begin(), explicit_operands.value().end());
        }

        for (auto [i, operand] : llvm::enumerate(operands)) {
            if (isa<qref::QuregType>(operand.getType())) {
                this->vQregResultIndices.push_back(existingNumResults + resIdx);
                this->rQregOperandIndices.push_back(i);
                resIdx++;
            }
            else if (isa<qref::QubitType>(operand.getType())) {
                if (this->tracker.isRootRQubit(operand)) {
                    this->rootVQubitResultIndices.push_back(existingNumResults + resIdx);
                    this->rootRQubitOperandIndices.push_back(i);
                }
                else {
                    this->nonRootVQubitResultIndices.push_back(existingNumResults + resIdx);
                    this->nonRootRQubitOperandIndices.push_back(i);
                }
                resIdx++;
            }
        }
    }

    /**
     * @brief Given an non-root rQubit Value, create a quantum.extract operation from the current
     * vQreg of the rQreg it belongs to. The extract op is created from the same index as the
     * rQubit's defining qref.get operation.
     */
    quantum::ExtractOp createExtractOp(Value rQubit)
    {
        assert(isa<qref::QubitType>(rQubit.getType()) && "Expected qref.bit type");
        auto getOp = dyn_cast<qref::GetOp>(rQubit.getDefiningOp());
        assert(getOp && "Only qref.get ops can produce qref.bit SSA values");

        OpBuilder::InsertionGuard guard(this->builder);

        std::optional<uint64_t> idxAttr = getOp.getIdxAttr();
        Value idxValue = getOp.getIdx();
        assert((idxAttr.has_value() ^ (idxValue != nullptr)) &&
               "expected exactly one index for extract op");

        Value vQreg = this->tracker.getCurrentVQreg(getOp.getQreg());

        Type qubitType = quantum::QubitType::get(rQubit.getContext());
        Type i64Type = this->builder.getI64Type();
        quantum::ExtractOp extractOp;
        if (idxAttr.has_value()) {
            extractOp = quantum::ExtractOp::create(this->builder, rQubit.getLoc(), qubitType, vQreg,
                                                   {}, IntegerAttr::get(i64Type, idxAttr.value()));
        }
        else {
            extractOp = quantum::ExtractOp::create(this->builder, rQubit.getLoc(), qubitType, vQreg,
                                                   idxValue, nullptr);
        }

        return extractOp;
    }
}; // struct TransientQubitExtractor

/**
 * @brief Given a reference semantics operation instance, migrate it to value semantics.
 *
 * We create the corresponding value semantics operation, with exactly the same operands and
 * attributes, except we replace the rQubit (and rQreg) Values with the corresponding current
 * vQubit (and vQreg) Values.
 *
 * If the vQubit Values do not exist in the IR yet, a quantum.extract op from the corresponding
 * quantum.reg is created, and the newly extracted quantum.bit Value is used.
 *
 * The result types of the value semantics operation will be the same as the old one, unless
 * explicitly overriden in the `newResultTypes` argument. When overridden, all new results types
 * that is a quantum.bit or quantum.reg will appear at the end of the results of the new operation.
 *
 * Any vQreg and root vQubit result Values from the newly created value semantics operation is
 * updated into the tracker. If no `newResultTypes` are provided, this update does not happen,
 * as none of the results from the old qref op could be a vQreg or vQubit.
 *
 * @tparam OpTy
 * @param builder
 * @param qrefOp
 * @param tracker
 * @param newResultTypes
 * @return OpTy
 */
template <typename OpTy>
OpTy migrateOpToValueSemantics(IRRewriter &builder, Operation *qrefOp, QubitValueTracker &tracker,
                               std::optional<TypeRange> newResultTypes)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(qrefOp);
    Location loc = qrefOp->getLoc();

    TransientQubitExtractor extractor(tracker, builder, qrefOp);

    // Create the new op using the generic state-based approach
    // We cannot just clone, since we are changing the op type
    OperationState state(loc, OpTy::getOperationName());

    SmallVector<Value> vOperands;
    for (Value v : qrefOp->getOperands()) {
        if (isa<qref::QubitType>(v.getType()) && tracker.isRootRQubit(v)) {
            vOperands.push_back(tracker.getCurrentVQubit(v));
        }
        else if (isa<qref::QuregType>(v.getType())) {
            vOperands.push_back(tracker.getCurrentVQreg(v));
        }
        else {
            // This branch includes classical operands and non-root rQubit operands
            vOperands.push_back(v);
        }
    }
    for (auto [vQubit, idx] : llvm::zip_equal(extractor.getExtractedVQubits(),
                                              extractor.getNonRootQubitOperandIndices())) {
        vOperands[idx] = vQubit;
    }

    state.addOperands(vOperands);
    state.addAttributes(qrefOp->getAttrs());

    SmallVector<Type> outTypes(qrefOp->getResultTypes());
    if (newResultTypes.has_value()) {
        outTypes.append(newResultTypes->begin(), newResultTypes->end());
    }
    state.addTypes(outTypes);

    Operation *newOp = builder.create(state);

    // Update tracker with results
    if (newResultTypes.has_value()) {
        for (auto [i, j] :
             llvm::zip_equal(extractor.getQRegOperandIndices(), extractor.getQRegResultIndices())) {
            tracker.setCurrentVQreg(qrefOp->getOperand(i), newOp->getResult(j));
        }

        for (auto [i, j] : llvm::zip_equal(extractor.getRootQubitOperandIndices(),
                                           extractor.getRootQubitResultIndices())) {
            tracker.setCurrentVQubit(qrefOp->getOperand(i), newOp->getResult(j));
        }
    }

    extractor.setVOp(newOp);

    return cast<OpTy>(newOp);
}

void _getNecessaryRegionRValuesImpl(Region &r, SetVector<Value> &necessaryRegionRValues,
                                    std::function<bool(Region &, Value)> isFromOutside)
{
    llvm::SmallDenseSet<Value, 8> rQregsTakenIn;

    r.walk([&](Operation *op) {
        if (!isa<qref::QRefDialect>(op->getDialect()) && !isa<func::CallOp>(op)) {
            return;
        }
        if (isa<qref::GetOp>(op)) {
            // qref.get is not a gate, do not count it as a user
            // For example, if the rQubit result from a qref.get has no users, the get op is not
            // actually needed by the region.
            return;
        }
        for (Value v : op->getOperands()) {
            if (isa<qref::QuregType>(v.getType())) {
                // Ignore allocations from inside the region itself
                if (isFromOutside(r, v)) {
                    necessaryRegionRValues.insert(v);
                    rQregsTakenIn.insert(v);
                }
            }
            else if (isa<qref::QubitType>(v.getType())) {
                if (auto getOp = v.getDefiningOp<qref::GetOp>()) {
                    Value rQreg = getRSourceRegisterValue(v);
                    if (isFromOutside(r, rQreg)) {
                        if (getOp.getIdx()) {
                            // dynamic extract index, must take in the reg
                            necessaryRegionRValues.insert(rQreg);
                            rQregsTakenIn.insert(rQreg);
                        }
                        else {
                            necessaryRegionRValues.insert(v);
                        }
                    }
                }
                else {
                    // Ignore allocations from inside the region itself
                    if (isFromOutside(r, v)) {
                        necessaryRegionRValues.insert(v);
                    }
                }
            }
        }
    });

    // If any rQregs are taken in, any rQubits belonging to them must not be taken in separately
    necessaryRegionRValues.remove_if([&](const Value &v) {
        if (isa<BlockArgument>(v)) {
            return false;
        }
        if (auto getOp = dyn_cast<qref::GetOp>(v.getDefiningOp())) {
            if (rQregsTakenIn.contains(getOp.getQreg())) {
                return true;
            }
        }
        return false;
    });

    // Remove aliasing get ops
    DenseSet<rQubitGetOpInfo> seenGetInfos;
    necessaryRegionRValues.remove_if([&](const Value &v) {
        if (!v.getDefiningOp<qref::GetOp>()) {
            return false;
        }

        rQubitGetOpInfo info = getGetOpInfo(v).value();
        // If already exists in set, insertion will fail, and we have seen an alias, so need to
        // remove
        return !seenGetInfos.insert(info).second;
    });
}

/**
 * @brief Collect the rQreg and rQubit Values that are captured into a region from above by closure.
 *
 * Reference semantics dialect operations do not take in or produce qreg Values, which means all
 * qreg Values are taken in via closure from above.
 *
 * When converting to value semantics, the vQregs and vQubits need to be taken in by the region-ed
 * operations explicitly.
 *
 * The collected rValues satisfy the following properties:
 * - If any rQubit Values are `qref.get`-ed from a dynamic index, the rQreg Value is collected
 * instead of the rQubit Value.
 * - If any rQreg Values are collected, none of the collected rQubit Values will be belonging to
 * the rQreg Values.
 * - All collected rQubit Values are guaranteed to not alias each other.
 *
 * Registers and qubits allocated within the region are not collected.
 *
 * @param r
 * @param necessaryRegionRValues
 */
void collectNecessaryRegionRValues(Region &r, SetVector<Value> &necessaryRegionRValues)
{
    _getNecessaryRegionRValuesImpl(r, necessaryRegionRValues, [&](Region &r, Value v) {
        return v.getParentRegion()->isProperAncestor(&r);
    });
}

/**
 * @brief Collect the rQreg and rQubit Values that are needed in a subroutine func op.
 *
 * The collected rValues satisfy the following properties:
 * - If any rQubit Values are `qref.get`-ed from a dynamic index, the rQreg Value is collected
 * instead of the rQubit Value.
 * - If any rQreg Values are collected, none of the collected rQubit Values will be belonging to
 * the rQreg Values.
 * - All collected rQubit Values are guaranteed to not alias each other.
 *
 * Registers and qubits allocated within the subroutine func op are not collected.
 *
 * @param f
 * @param necessarySubroutineRValues
 */
void collectNecessarySubroutineRValues(func::FuncOp f, SetVector<Value> &necessarySubroutineRValues)
{
    _getNecessaryRegionRValuesImpl(
        f.getBody(), necessarySubroutineRValues,
        [&](Region &r, Value v) { return llvm::is_contained(r.getArguments(), v); });
}

/**
 * @brief An info object to store what new arguments the converted subroutine needs.
 *
 * The strategy to convert subroutines and calls are as follows:
 *
 * The subroutine body is walked over, and the necessary rValues are collected.
 * An rValue is deemed necessary if it is an operand to a gate-like operation inside the subroutine,
 * and does not belong to an allocation from inside the subroutine.
 * These are the values that need to be passed in from outside the subroutine.
 *
 * Of the necessary rValues, there will be 2 kinds:
 * - Either it is already a subroutine argument (A);
 * - or, it is a rQubit from a getOp inside the subroutine, whose rQreg is a subroutine argument,
 * and whose extract index is static (B)
 *
 * For each of these collected necessary rValues, an entry is added to this info object.
 * - For type A, the entry is a single number, indicating the argument index
 * - For type B, the entry is a pair of numbers, indicating the argument index of the rQreg, and the
 * static extract index
 *
 * For example, consider the subroutine and the call (pseudocode)
 *
 *    func.func @subroutine(%r: !qref.reg<3>, %q: !qref.bit, %param: f64) -> () {
 *        %q0 = qref.get %r[0]
 *        %q1 = qref.get %r[1]
 *
 *        %r_inside = qref.alloc(2)
 *        %q_inside = qref.get %r_inside[1]
 *
 *        qref.custom "gate"(%param) %q0, %q1, %q, %q_inside
 *        return
 *    }
 *
 *    func.call @subroutine(%r_call, %q_call, %param_call) : (!qref.reg<3>, !qref.bit, f64) -> ()
 *
 * The necessary rValues of the subroutine are %q0 (type B), %q1 (type B) and %q (type A)
 * The content of the info object would be
 *   [[0, 0], [0, 1], 1]
 *
 * The purpose is so that when building the new call op, extract ops can be properly built before
 * the new call. The new call would be
 *    %q0_call = qref.get %r_call[0]    // from reg = call_old_args[0], extract idx = 0
 *    %q1_call = qref.get %r_call[1]    // from reg = call_old_args[0], extract idx = 1
 *    func.call @subroutine(%param_call, %q0_call, %q1_call, %q_call)  // old_call_args[1] = %q_call
 *
 * All new args, one for each rValue needed by the subroutine, must be appended to the end of
 * the list of new arguments
 * All old qref args must be purged from the call, since the newly collected necessary rValues are a
 * complete source of truth.
 *
 * Note that this object performs no IR mutation whatsoever. It is only an analysis.
 * It is to be used by the handlers of call ops, to build the new call op signature.
 */
struct SubroutineInfo {
  public:
    SubroutineInfo(func::FuncOp f) : subroutine(f)
    {
        collectNecessarySubroutineRValues(this->subroutine, this->necessarySubroutineRValues);
        for (auto rValue : this->necessarySubroutineRValues) {
            if (auto rValueAsArg = dyn_cast<BlockArgument>(rValue)) {
                newArgsInfo.push_back(rValueAsArg.getArgNumber());
                continue;
            }
            auto getOp = dyn_cast<qref::GetOp>(rValue.getDefiningOp());
            assert(getOp && "Gates inside a subroutine in reference semantics must act on either "
                            "qref arguments of the subroutine, allocations from within the "
                            "subroutine, or qref.bit values produced by qref.get ops");
            Value rQreg = getOp.getQreg();
            assert(llvm::is_contained(f.getArguments(), rQreg) &&
                   "Subroutines in reference semantics cannot take in qref.bit values that are not "
                   "from a single qubit allocation");
            assert(!getOp.getIdx() && "qref.bit values from a get op inside a subroutine "
                                      "scheduled to be passed in as "
                                      "an quantum.extract-ed qubit from the call site must "
                                      "have static extract index");
            unsigned argnum = cast<BlockArgument>(rQreg).getArgNumber();
            this->newArgsInfo.push_back(std::make_pair(argnum, getOp.getIdxAttr().value()));
        }
    }

    const SetVector<Value> &getNecessarySubroutineRValues()
    {
        return this->necessarySubroutineRValues;
    }

    const SmallVector<std::variant<unsigned, std::pair<unsigned, uint64_t>>> &getNewArgsInfo()
    {
        return this->newArgsInfo;
    }

  private:
    func::FuncOp subroutine;
    SetVector<Value> necessarySubroutineRValues;
    SmallVector<std::variant<unsigned, std::pair<unsigned, uint64_t>>> newArgsInfo;
}; // struct SubroutineInfo

void stageCallOpForConversion(IRRewriter &builder, func::CallOp callOp,
                              SubroutineInfo &subroutineInfo)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = callOp->getContext();
    Location loc = callOp->getLoc();

    builder.setInsertionPoint(callOp);
    SmallVector<Value> newCallArgs;
    ValueRange oldCallArgs(callOp->getOperands());
    for (Value oldCallArg : oldCallArgs) {
        if (!isa<qref::QubitType, qref::QuregType>(oldCallArg.getType())) {
            newCallArgs.push_back(oldCallArg);
        }
    }
    for (auto newArgsInfo : subroutineInfo.getNewArgsInfo()) {
        if (std::holds_alternative<unsigned>(newArgsInfo)) {
            newCallArgs.push_back(oldCallArgs[std::get<unsigned>(newArgsInfo)]);
        }
        else {
            auto [oldCallArgIdx, extractIdx] = std::get<std::pair<unsigned, uint64_t>>(newArgsInfo);
            assert(isa<qref::QuregType>(oldCallArgs[oldCallArgIdx].getType()) && "Expected rQreg");
            auto getOp = qref::GetOp::create(builder, loc, qref::QubitType::get(ctx),
                                             oldCallArgs[oldCallArgIdx], nullptr,
                                             IntegerAttr::get(builder.getI64Type(), extractIdx));
            newCallArgs.push_back(getOp.getQubit());
        }
    }
    auto newCallOp = func::CallOp::create(builder, loc, callOp->getResultTypes(),
                                          callOp.getCallee(), newCallArgs);
    builder.replaceOp(callOp, newCallOp);
}

void handleAlloc(IRRewriter &builder, qref::AllocOp rAllocOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rAllocOp);

    Location loc = rAllocOp.getLoc();
    MLIRContext *ctx = rAllocOp.getContext();
    Type qregType = quantum::QuregType::get(ctx);
    Type i64Type = builder.getI64Type();

    quantum::AllocOp vAllocOp;
    std::optional<uint64_t> nqubitsAttr = rAllocOp.getNqubitsAttr();
    if (nqubitsAttr.has_value()) {
        vAllocOp = quantum::AllocOp::create(builder, loc, qregType, {},
                                            IntegerAttr::get(i64Type, *nqubitsAttr));
    }
    else {
        vAllocOp = quantum::AllocOp::create(builder, loc, qregType, rAllocOp.getNqubits(), nullptr);
    }
    tracker.setCurrentVQreg(rAllocOp.getQreg(), vAllocOp.getQreg());
}

void handleDealloc(IRRewriter &builder, qref::DeallocOp rDeallocOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rDeallocOp);
    Location loc = rDeallocOp.getLoc();

    quantum::DeallocOp::create(builder, loc, tracker.getCurrentVQreg(rDeallocOp.getQreg()));

    builder.eraseOp(rDeallocOp);
}

void handleAllocQubit(IRRewriter &builder, qref::AllocQubitOp rAllocQbOp,
                      QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rAllocQbOp);
    Location loc = rAllocQbOp.getLoc();

    auto vAllocQbOp = quantum::AllocQubitOp::create(builder, loc);
    tracker.setCurrentVQubit(rAllocQbOp.getQubit(), vAllocQbOp.getQubit());
}

void handleDeallocQubit(IRRewriter &builder, qref::DeallocQubitOp rDeallocQbOp,
                        QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rDeallocQbOp);
    Location loc = rDeallocQbOp.getLoc();

    quantum::DeallocQubitOp::create(builder, loc,
                                    tracker.getCurrentVQubit(rDeallocQbOp.getQubit()));

    builder.eraseOp(rDeallocQbOp);
}

void handleGate(IRRewriter &builder, qref::QuantumOperation rGateOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = rGateOp.getContext();

    SmallVector<Type> qubitResultsType;
    for (size_t i = 0; i < rGateOp.getQubitOperands().size(); i++) {
        qubitResultsType.push_back(quantum::QubitType::get(ctx));
    }

    quantum::QuantumOperation vGateOp;
    Operation *_rGateOp = rGateOp.getOperation();
    if (auto rCustomOp = dyn_cast<qref::CustomOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::CustomOp>(builder, rCustomOp, tracker,
                                                               qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rCustomOp));
    }
    else if (auto rPauliRotOP = dyn_cast<qref::PauliRotOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PauliRotOp>(builder, rPauliRotOP, tracker,
                                                                 qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rPauliRotOP));
    }
    else if (auto rGPhaseOp = dyn_cast<qref::GlobalPhaseOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::GlobalPhaseOp>(builder, rGPhaseOp, tracker,
                                                                    qubitResultsType);
    }
    else if (auto rMultiRZOp = dyn_cast<qref::MultiRZOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::MultiRZOp>(builder, rMultiRZOp, tracker,
                                                                qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rMultiRZOp));
    }
    else if (auto rPCPhaseOp = dyn_cast<qref::PCPhaseOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::PCPhaseOp>(builder, rPCPhaseOp, tracker,
                                                                qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, rPCPhaseOp));
    }
    else if (auto qQubitUnitaryOp = dyn_cast<qref::QubitUnitaryOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::QubitUnitaryOp>(builder, qQubitUnitaryOp,
                                                                     tracker, qubitResultsType);
        vGateOp->setAttr("resultSegmentSizes", getResultSegmentSizes(builder, qQubitUnitaryOp));
    }
    else if (auto rSetStateOp = dyn_cast<qref::SetStateOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::SetStateOp>(builder, rSetStateOp, tracker,
                                                                 qubitResultsType);
    }
    else if (auto rSetBasisStateOp = dyn_cast<qref::SetBasisStateOp>(_rGateOp)) {
        vGateOp = migrateOpToValueSemantics<quantum::SetBasisStateOp>(builder, rSetBasisStateOp,
                                                                      tracker, qubitResultsType);
    }
    else {
        rGateOp->emitOpError("unknown gate op in qref dialect");
    }

    builder.eraseOp(rGateOp);
}

void handleMeasure(IRRewriter &builder, qref::MeasureOp rMeasureOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = rMeasureOp.getContext();

    auto vMeasureOp = migrateOpToValueSemantics<quantum::MeasureOp>(builder, rMeasureOp, tracker,
                                                                    {quantum::QubitType::get(ctx)});
    builder.replaceAllUsesWith(rMeasureOp.getMres(), vMeasureOp.getMres());
    builder.eraseOp(rMeasureOp);
}

void handleMeasureInBasis(IRRewriter &builder, qref::MeasureInBasisOp rMeasureInBasisOp,
                          QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = rMeasureInBasisOp.getContext();

    auto vMeasureOp = migrateOpToValueSemantics<mbqc::MeasureInBasisOp>(
        builder, rMeasureInBasisOp, tracker, {quantum::QubitType::get(ctx)});
    builder.replaceAllUsesWith(rMeasureInBasisOp.getMres(), vMeasureOp.getMres());
    builder.eraseOp(rMeasureInBasisOp);
}

void handleCall(IRRewriter &builder, func::CallOp callOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    MLIRContext *ctx = callOp.getContext();

    SmallVector<Type> newResultTypes;
    for (auto callArg : callOp.getOperands()) {
        if (isa<qref::QuregType>(callArg.getType())) {
            newResultTypes.push_back(quantum::QuregType::get(ctx));
        }
        else if (isa<qref::QubitType>(callArg.getType())) {
            newResultTypes.push_back(quantum::QubitType::get(ctx));
        }
    }

    if (newResultTypes.size() == 0) {
        return;
    }

    auto newCallOp =
        migrateOpToValueSemantics<func::CallOp>(builder, callOp, tracker, newResultTypes);

    for (auto [i, v] : llvm::enumerate(callOp->getResults())) {
        builder.replaceAllUsesWith(v, newCallOp->getResult(i));
    }

    builder.eraseOp(callOp);
}

void handleCompbasis(IRRewriter &builder, qref::ComputationalBasisOp rCompbasisOp,
                     QubitValueTracker &tracker)
{
    auto vCompbasisOp =
        migrateOpToValueSemantics<quantum::ComputationalBasisOp>(builder, rCompbasisOp, tracker);
    builder.replaceOp(rCompbasisOp, vCompbasisOp);
}

void handleNamedObs(IRRewriter &builder, qref::NamedObsOp rNamedObsOp, QubitValueTracker &tracker)
{
    auto vNamedObsOp =
        migrateOpToValueSemantics<quantum::NamedObsOp>(builder, rNamedObsOp, tracker);
    builder.replaceOp(rNamedObsOp, vNamedObsOp);
}

void handleHermitian(IRRewriter &builder, qref::HermitianOp rHermitianOp,
                     QubitValueTracker &tracker)
{
    auto vHermitianOp =
        migrateOpToValueSemantics<quantum::HermitianOp>(builder, rHermitianOp, tracker);
    builder.replaceOp(rHermitianOp, vHermitianOp);
}

void handleGraphStatePrep(IRRewriter &builder, qref::GraphStatePrepOp rGraphStatePrepOp,
                          QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rGraphStatePrepOp);

    Location loc = rGraphStatePrepOp.getLoc();
    MLIRContext *ctx = rGraphStatePrepOp.getContext();
    Type qregType = quantum::QuregType::get(ctx);

    auto vGraphStatePrepOp = mbqc::GraphStatePrepOp::create(
        builder, loc, qregType, rGraphStatePrepOp.getAdjMatrix(), rGraphStatePrepOp.getInitOp(),
        rGraphStatePrepOp.getEntangleOp());

    tracker.setCurrentVQreg(rGraphStatePrepOp.getQreg(), vGraphStatePrepOp.getQreg());
}

/**
 * @brief Append the current root vQreg and vQubit values to the terminator operation of a region,
 * with the root rQreg and rQubit values being the ones passed in in `rValuesToReturn`.
 *
 * This is necessary since qref operations do not return quantum values.
 *
 * @param r
 * @param rValuesToReturn
 * @param tracker
 */
void addRootVValuesToRetOp(Operation *retOp, ArrayRef<Value> rValuesToReturn,
                           QubitValueTracker &tracker)
{
    SmallVector<Value> retVals(retOp->getOperands());
    for (auto rValue : rValuesToReturn) {
        if (isa<qref::QubitType>(rValue.getType())) {
            retVals.push_back(tracker.getCurrentVQubit(rValue));
        }
        else if (isa<qref::QuregType>(rValue.getType())) {
            retVals.push_back(tracker.getCurrentVQreg(rValue));
        }
    }
    retOp->setOperands(retVals);
}

/**
 * @brief Add value semantics arguments to the region, one for each rValue used by the region, and
 * handle the region. The newly added arguments are all considered root for the region.
 *
 * @param builder
 * @param rValuesUsedByRegion
 * @param r
 */
void addVArgsToRegionAndHandle(IRRewriter &builder, const SetVector<Value> &rValuesUsedByRegion,
                               Region &r)
{
    MLIRContext *ctx = r.getContext();
    Location loc = r.getLoc();

    QubitValueTracker regionTracker;
    for (auto rValue : rValuesUsedByRegion) {
        Value newArg;
        if (isa<qref::QubitType>(rValue.getType())) {
            newArg = r.front().addArgument(quantum::QubitType::get(ctx), loc);
            regionTracker.setCurrentVQubit(rValue, newArg);
        }
        else if (isa<qref::QuregType>(rValue.getType())) {
            newArg = r.front().addArgument(quantum::QuregType::get(ctx), loc);
            regionTracker.setCurrentVQreg(rValue, newArg);
        }
    }

    handleRegion(builder, r, regionTracker);
    addRootVValuesToRetOp(r.front().getTerminator(), rValuesUsedByRegion.getArrayRef(),
                          regionTracker);
}

/**
 * @brief Handle the region with all current vValues kept alive from above, and with the extracted
 * vQubits from the extractor treated as root qubit values for the region.
 *
 * This method is needed to handle scf.if and scf.index_switch, where an explicit list of operands
 * is not available.
 *
 * @param builder
 * @param rValuesUsedByRegion
 * @param extractor
 * @param outerTracker
 * @param r
 */
void addPretendedArgsToRegionAndHandle(IRRewriter &builder,
                                       const SetVector<Value> &rValuesUsedByRegion,
                                       TransientQubitExtractor &extractor,
                                       QubitValueTracker &outerTracker, Region &r)
{
    // Copy over all the existing root Value maps in the outer scope
    QubitValueTracker regionTracker = outerTracker;

    // newly extracted qubits, though non-root outside, are considered root inside!
    for (auto [vQubit, idx] : llvm::zip_equal(extractor.getExtractedVQubits(),
                                              extractor.getNonRootQubitOperandIndices())) {
        regionTracker.setCurrentVQubit(rValuesUsedByRegion[idx], vQubit);
    }

    handleRegion(builder, r, regionTracker);
    addRootVValuesToRetOp(r.front().getTerminator(), rValuesUsedByRegion.getArrayRef(),
                          regionTracker);
}

void handleAdjoint(IRRewriter &builder, qref::AdjointOp rAdjointOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(rAdjointOp);
    Location loc = rAdjointOp->getLoc();

    SetVector<Value> rValuesUsedByRegion;
    collectNecessaryRegionRValues(rAdjointOp.getRegion(), rValuesUsedByRegion);

    quantum::AdjointOp vAdjointOp;
    {
        // 1. Send in the vValues from above as arguments
        // This handles the flow from outside the vAdjointOp into the vAdjointOp
        // i.e. the extract op results are sent in as operands to the vAdjointOp
        SmallVector<Value> regionOperands;
        regionOperands.append(rValuesUsedByRegion.begin(), rValuesUsedByRegion.end());
        TransientQubitExtractor extractor(tracker, builder, rAdjointOp, regionOperands);
        SmallVector<Value> vAdjointOperands;

        for (Value rValue : rValuesUsedByRegion) {
            if (isa<qref::QubitType>(rValue.getType()) && tracker.isRootRQubit(rValue)) {
                vAdjointOperands.push_back(tracker.getCurrentVQubit(rValue));
            }
            else if (isa<qref::QuregType>(rValue.getType())) {
                vAdjointOperands.push_back(tracker.getCurrentVQreg(rValue));
            }
            else {
                // To be replaced with extracted vQubits
                vAdjointOperands.push_back(rValue);
            }
        }
        for (auto [vQubit, idx] : llvm::zip_equal(extractor.getExtractedVQubits(),
                                                  extractor.getNonRootQubitOperandIndices())) {
            vAdjointOperands[idx] = vQubit;
        }

        vAdjointOp =
            quantum::AdjointOp::create(builder, loc, TypeRange(vAdjointOperands), vAdjointOperands);

        // 2. Move operations from old body to new body
        builder.inlineRegionBefore(rAdjointOp.getRegion(), vAdjointOp.getRegion(),
                                   vAdjointOp.getRegion().end());
        builder.setInsertionPointToEnd(&vAdjointOp.getRegion().front());
        quantum::YieldOp::create(builder, loc, {});

        // 3. Create new args with quantum.bit/reg types and set them as root for the new region
        addVArgsToRegionAndHandle(builder, rValuesUsedByRegion, vAdjointOp.getRegion());

        // Update tracker with results
        for (auto [i, j] :
             llvm::zip_equal(extractor.getQRegOperandIndices(), extractor.getQRegResultIndices())) {
            tracker.setCurrentVQreg(rValuesUsedByRegion[i], vAdjointOp->getResult(j));
        }

        for (auto [i, j] : llvm::zip_equal(extractor.getRootQubitOperandIndices(),
                                           extractor.getRootQubitResultIndices())) {
            tracker.setCurrentVQubit(rValuesUsedByRegion[i], vAdjointOp->getResult(j));
        }

        extractor.setVOp(vAdjointOp);
    }

    builder.eraseOp(rAdjointOp);
}

void handleCtrl(IRRewriter &builder, qref::CtrlOp rCtrlOp, QubitValueTracker &tracker)
{
     // TODO
}


void createElseBranchWithDefaultYields(IRRewriter &builder,
                                       const SetVector<Value> &rValuesUsedByRegion,
                                       TransientQubitExtractor &extractor,
                                       QubitValueTracker &outerTracker, scf::IfOp ifOp)
{
    // no explicit "else" region on the original if op, just yield whatever the closure
    // variables were from the outer scope, and yield the extracted qubits from before entering the
    // ifOp directly
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Value> elseYieldVals;
    for (Value v : rValuesUsedByRegion) {
        if (isa<qref::QuregType>(v.getType())) {
            elseYieldVals.push_back(outerTracker.getCurrentVQreg(v));
        }
        else if (isa<qref::QubitType>(v.getType()) && outerTracker.isRootRQubit(v)) {
            elseYieldVals.push_back(outerTracker.getCurrentVQubit(v));
        }
        else {
            elseYieldVals.push_back(v);
        }
    }
    for (auto [extractedVQubit, idx] : llvm::zip_equal(extractor.getExtractedVQubits(),
                                                       extractor.getNonRootQubitOperandIndices())) {
        elseYieldVals[idx] = extractedVQubit;
    }
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    scf::YieldOp::create(builder, ifOp->getLoc(), elseYieldVals);
}

void handleIf(IRRewriter &builder, scf::IfOp ifOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(ifOp);
    MLIRContext *ctx = ifOp->getContext();
    Location loc = ifOp->getLoc();
    bool hasElseRegion = !ifOp.getElseRegion().empty();

    SetVector<Value> rValuesUsedByRegion;
    collectNecessaryRegionRValues(ifOp.getThenRegion(), rValuesUsedByRegion);
    if (hasElseRegion) {
        collectNecessaryRegionRValues(ifOp.getElseRegion(), rValuesUsedByRegion);
    }

    if (rValuesUsedByRegion.size() == 0) {
        return;
    }

    scf::IfOp newIfOp;
    {
        // 1. Send in the vValues from above as arguments
        // This handles the flow from outside the new if op into the new if op
        // i.e. the extract op results are viewed as the new if op's "operands"
        // Note that since scf.if cannot formally take operands, we need to pretend the
        // extracted vQubits are "root".
        SmallVector<Value> regionOperands;
        regionOperands.append(rValuesUsedByRegion.begin(), rValuesUsedByRegion.end());
        TransientQubitExtractor extractor(tracker, builder, ifOp, regionOperands);

        SmallVector<Type> newResultTypes(ifOp->getResultTypes());
        for (Value rValue : rValuesUsedByRegion) {
            if (isa<qref::QubitType>(rValue.getType())) {
                newResultTypes.push_back(quantum::QubitType::get(ctx));
            }
            else if (isa<qref::QuregType>(rValue.getType())) {
                newResultTypes.push_back(quantum::QuregType::get(ctx));
            }
        }

        // scf.if op always requires an else block if returning any results
        newIfOp = scf::IfOp::create(builder, loc, newResultTypes, ifOp.getCondition(),
                                    /*withElseRegion=*/true);

        // 2. Handle the "then" region
        builder.eraseBlock(newIfOp.thenBlock());
        builder.inlineRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                   newIfOp.getThenRegion().end());
        addPretendedArgsToRegionAndHandle(builder, rValuesUsedByRegion, extractor, tracker,
                                          newIfOp.getThenRegion());

        // 3. Handle "else" region
        // If none existed before, we need to create an empty "else" region, just for the yield
        // structure demanded by scf.if op
        if (hasElseRegion) {
            builder.eraseBlock(newIfOp.elseBlock());
            builder.inlineRegionBefore(ifOp.getElseRegion(), newIfOp.getElseRegion(),
                                       newIfOp.getElseRegion().end());
            addPretendedArgsToRegionAndHandle(builder, rValuesUsedByRegion, extractor, tracker,
                                              newIfOp.getElseRegion());
        }
        else {
            // no explicit "else" region on the original if op, just yield whatever the closure
            // variables were, and yield the extracted qubits from before entering the ifOp directly
            createElseBranchWithDefaultYields(builder, rValuesUsedByRegion, extractor, tracker,
                                              newIfOp);
        }

        // Update outer tracker with results
        for (auto [i, j] :
             llvm::zip_equal(extractor.getQRegOperandIndices(), extractor.getQRegResultIndices())) {
            tracker.setCurrentVQreg(rValuesUsedByRegion[i], newIfOp->getResult(j));
        }

        for (auto [i, j] : llvm::zip_equal(extractor.getRootQubitOperandIndices(),
                                           extractor.getRootQubitResultIndices())) {
            tracker.setCurrentVQubit(rValuesUsedByRegion[i], newIfOp->getResult(j));
        }

        extractor.setVOp(newIfOp);
    }

    for (auto [i, v] : llvm::enumerate(ifOp->getResults())) {
        builder.replaceAllUsesWith(v, newIfOp->getResult(i));
    }

    builder.eraseOp(ifOp);
}

void handleSwitch(IRRewriter &builder, scf::IndexSwitchOp switchOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(switchOp);
    MLIRContext *ctx = switchOp->getContext();
    Location loc = switchOp->getLoc();

    SetVector<Value> rValuesUsedByRegion;
    for (Region &r : switchOp.getCaseRegions()) {
        collectNecessaryRegionRValues(r, rValuesUsedByRegion);
    }
    collectNecessaryRegionRValues(switchOp.getDefaultRegion(), rValuesUsedByRegion);

    if (rValuesUsedByRegion.size() == 0) {
        return;
    }

    scf::IndexSwitchOp newSwitchOp;
    {
        // 1. Send in the vValues from above as arguments
        // This handles the flow from outside the new if op into the new switch op
        // i.e. the extract op results are viewed as the new switch op's "operands"
        // Note that since scf.index_switch cannot formally take operands, we need to pretend the
        // extracted vQubits are "root".
        SmallVector<Value> regionOperands;
        regionOperands.append(rValuesUsedByRegion.begin(), rValuesUsedByRegion.end());
        TransientQubitExtractor extractor(tracker, builder, switchOp, regionOperands);

        SmallVector<Type> newResultTypes(switchOp->getResultTypes());
        for (Value rValue : rValuesUsedByRegion) {
            if (isa<qref::QubitType>(rValue.getType())) {
                newResultTypes.push_back(quantum::QubitType::get(ctx));
            }
            else if (isa<qref::QuregType>(rValue.getType())) {
                newResultTypes.push_back(quantum::QuregType::get(ctx));
            }
        }

        newSwitchOp = scf::IndexSwitchOp::create(builder, loc, newResultTypes, switchOp.getArg(),
                                                 switchOp.getCases(), switchOp.getNumCases());

        // 2. Handle the "default" region
        builder.inlineRegionBefore(switchOp.getDefaultRegion(), newSwitchOp.getDefaultRegion(),
                                   newSwitchOp.getDefaultRegion().end());
        addPretendedArgsToRegionAndHandle(builder, rValuesUsedByRegion, extractor, tracker,
                                          newSwitchOp.getDefaultRegion());

        // 3. Handle the case regions
        for (auto [oldCaseRegion, newCaseRegion] :
             llvm::zip_equal(switchOp.getCaseRegions(), newSwitchOp.getCaseRegions())) {
            builder.inlineRegionBefore(oldCaseRegion, newCaseRegion, newCaseRegion.end());
            addPretendedArgsToRegionAndHandle(builder, rValuesUsedByRegion, extractor, tracker,
                                              newCaseRegion);
        }

        // Update outer tracker with results
        for (auto [i, j] :
             llvm::zip_equal(extractor.getQRegOperandIndices(), extractor.getQRegResultIndices())) {
            tracker.setCurrentVQreg(rValuesUsedByRegion[i], newSwitchOp->getResult(j));
        }

        for (auto [i, j] : llvm::zip_equal(extractor.getRootQubitOperandIndices(),
                                           extractor.getRootQubitResultIndices())) {
            tracker.setCurrentVQubit(rValuesUsedByRegion[i], newSwitchOp->getResult(j));
        }

        extractor.setVOp(newSwitchOp);
    }

    for (auto [i, v] : llvm::enumerate(switchOp->getResults())) {
        builder.replaceAllUsesWith(v, newSwitchOp->getResult(i));
    }

    builder.eraseOp(switchOp);
}

void handleFor(IRRewriter &builder, scf::ForOp forOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(forOp);
    Location loc = forOp->getLoc();

    SetVector<Value> rValuesUsedByRegion;
    collectNecessaryRegionRValues(forOp.getRegion(), rValuesUsedByRegion);

    if (rValuesUsedByRegion.size() == 0) {
        return;
    }

    scf::ForOp newLoop;
    {
        // 1. Send in the vValues from above as arguments
        // This handles the flow from outside the new loop into the new loop
        // i.e. the extract op results are sent in as operands to the new for loop
        SmallVector<Value> regionOperands(forOp.getRegion().front().getArguments());
        regionOperands.append(rValuesUsedByRegion.begin(), rValuesUsedByRegion.end());
        TransientQubitExtractor extractor(tracker, builder, forOp, regionOperands);
        SmallVector<Value> newIterArgs(forOp.getInitArgs());

        for (Value rValue : rValuesUsedByRegion) {
            if (isa<qref::QubitType>(rValue.getType()) && tracker.isRootRQubit(rValue)) {
                newIterArgs.push_back(tracker.getCurrentVQubit(rValue));
            }
            else if (isa<qref::QuregType>(rValue.getType())) {
                newIterArgs.push_back(tracker.getCurrentVQreg(rValue));
            }
            else {
                // To be replaced with extracted vQubits
                newIterArgs.push_back(rValue);
            }
        }
        for (auto [vQubit, idx] : llvm::zip_equal(extractor.getExtractedVQubits(),
                                                  extractor.getNonRootQubitOperandIndices())) {
            // The loop's block always takes in the iteration variable (the `i`) as a block argument
            newIterArgs[idx - 1] = vQubit;
        }

        newLoop = scf::ForOp::create(builder, loc, forOp.getLowerBound(), forOp.getUpperBound(),
                                     forOp.getStep(), newIterArgs);

        // 2. Move operations from old body to new body
        builder.eraseBlock(newLoop.getBody());
        builder.inlineRegionBefore(forOp.getRegion(), newLoop.getRegion(),
                                   newLoop.getRegion().end());

        // 3. Create new args with quantum.bit/reg types and set them as root for the new region
        addVArgsToRegionAndHandle(builder, rValuesUsedByRegion, newLoop.getRegion());

        // Update tracker with results
        // Again, The loop's block always takes in the iteration variable (the `i`) as a block
        // argument
        for (auto [i, j] :
             llvm::zip_equal(extractor.getQRegOperandIndices(), extractor.getQRegResultIndices())) {
            tracker.setCurrentVQreg(rValuesUsedByRegion[i - 1 - forOp->getNumResults()],
                                    newLoop->getResult(j));
        }

        for (auto [i, j] : llvm::zip_equal(extractor.getRootQubitOperandIndices(),
                                           extractor.getRootQubitResultIndices())) {
            tracker.setCurrentVQubit(rValuesUsedByRegion[i - 1 - forOp->getNumResults()],
                                     newLoop->getResult(j));
        }

        extractor.setVOp(newLoop);
    }

    for (auto [i, v] : llvm::enumerate(forOp->getResults())) {
        builder.replaceAllUsesWith(v, newLoop->getResult(i));
    }

    builder.eraseOp(forOp);
}

void handleWhile(IRRewriter &builder, scf::WhileOp whileOp, QubitValueTracker &tracker)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(whileOp);
    Location loc = whileOp->getLoc();
    MLIRContext *ctx = whileOp.getContext();

    SetVector<Value> rValuesUsedByRegion;
    collectNecessaryRegionRValues(whileOp.getBefore(), rValuesUsedByRegion);
    collectNecessaryRegionRValues(whileOp.getAfter(), rValuesUsedByRegion);

    if (rValuesUsedByRegion.size() == 0) {
        return;
    }

    unsigned numExistingOperands = whileOp.getBefore().front().getArguments().size();

    scf::WhileOp newLoop;
    {
        // 1. Send in the vValues from above as arguments
        // This handles the flow from outside the new loop into the new loop
        // i.e. the extract op results are sent in as operands to the new loop
        SmallVector<Value> regionOperands(whileOp.getBefore().front().getArguments());
        regionOperands.append(rValuesUsedByRegion.begin(), rValuesUsedByRegion.end());
        TransientQubitExtractor extractor(tracker, builder, whileOp, regionOperands);
        SmallVector<Value> newIterArgs(whileOp.getInits());
        SmallVector<Type> newResultTypes(whileOp->getResultTypes());

        for (Value rValue : rValuesUsedByRegion) {
            if (isa<qref::QubitType>(rValue.getType())) {
                newResultTypes.push_back(quantum::QubitType::get(ctx));
                if (tracker.isRootRQubit(rValue)) {
                    newIterArgs.push_back(tracker.getCurrentVQubit(rValue));
                }
                else {
                    // To be replaced with extracted vQubits
                    newIterArgs.push_back(rValue);
                }
            }
            else if (isa<qref::QuregType>(rValue.getType())) {
                newIterArgs.push_back(tracker.getCurrentVQreg(rValue));
                newResultTypes.push_back(quantum::QuregType::get(ctx));
            }
        }
        for (auto [vQubit, idx] : llvm::zip_equal(extractor.getExtractedVQubits(),
                                                  extractor.getNonRootQubitOperandIndices())) {
            newIterArgs[idx] = vQubit;
        }

        newLoop = scf::WhileOp::create(builder, loc, newResultTypes, newIterArgs);

        // 2. Move operations from old body to new body
        builder.inlineRegionBefore(whileOp.getBefore(), newLoop.getBefore(),
                                   newLoop.getBefore().end());
        builder.inlineRegionBefore(whileOp.getAfter(), newLoop.getAfter(),
                                   newLoop.getAfter().end());

        // 3. Create new args with quantum.bit/reg types and set them as root for the new region
        addVArgsToRegionAndHandle(builder, rValuesUsedByRegion, newLoop.getBefore());
        addVArgsToRegionAndHandle(builder, rValuesUsedByRegion, newLoop.getAfter());

        // Update tracker with results
        for (auto [i, j] :
             llvm::zip_equal(extractor.getQRegOperandIndices(), extractor.getQRegResultIndices())) {
            tracker.setCurrentVQreg(rValuesUsedByRegion[i - numExistingOperands],
                                    newLoop->getResult(j));
        }

        for (auto [i, j] : llvm::zip_equal(extractor.getRootQubitOperandIndices(),
                                           extractor.getRootQubitResultIndices())) {
            tracker.setCurrentVQubit(rValuesUsedByRegion[i - numExistingOperands],
                                     newLoop->getResult(j));
        }

        extractor.setVOp(newLoop);
    }

    for (auto [i, v] : llvm::enumerate(whileOp->getResults())) {
        builder.replaceAllUsesWith(v, newLoop->getResult(i));
    }

    builder.eraseOp(whileOp);
}

void handleSubroutine(IRRewriter &builder, func::FuncOp f,
                      const SetVector<Value> &rValuesUsedBySubroutine)
{
    MLIRContext *ctx = f.getContext();
    OpBuilder::InsertionGuard guard(builder);
    Location loc = f->getLoc();

    // Add new quantum arguments
    QubitValueTracker regionTracker;
    for (auto rValue : rValuesUsedBySubroutine) {
        Value newArg;
        if (isa<qref::QubitType>(rValue.getType())) {
            newArg = f.getBody().front().addArgument(quantum::QubitType::get(ctx), loc);
            regionTracker.setCurrentVQubit(rValue, newArg);
        }
        else if (isa<qref::QuregType>(rValue.getType())) {
            newArg = f.getBody().front().addArgument(quantum::QuregType::get(ctx), loc);
            regionTracker.setCurrentVQreg(rValue, newArg);
        }
    }

    handleRegion(builder, f.getBody(), regionTracker);
    addRootVValuesToRetOp(f.front().getTerminator(), rValuesUsedBySubroutine.getArrayRef(),
                          regionTracker);

    eraseAllRemainingAnchorRValues(f);

    // Nuke all old qref arguments
    f.front().eraseArguments(
        [](BlockArgument arg) { return isa<qref::QubitType, qref::QuregType>(arg.getType()); });

    f.setFunctionType(FunctionType::get(ctx, f.front().getArgumentTypes(),
                                        f.front().getTerminator()->getOperandTypes()));
}

void handleRegion(IRRewriter &builder, Region &r, QubitValueTracker &tracker)
{
    r.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (auto rAllocOp = dyn_cast<qref::AllocOp>(op)) {
            handleAlloc(builder, rAllocOp, tracker);
        }
        else if (auto rDeallocOp = dyn_cast<qref::DeallocOp>(op)) {
            handleDealloc(builder, rDeallocOp, tracker);
        }
        else if (auto rAllocQbOp = dyn_cast<qref::AllocQubitOp>(op)) {
            handleAllocQubit(builder, rAllocQbOp, tracker);
        }
        else if (auto rDeallocQbOp = dyn_cast<qref::DeallocQubitOp>(op)) {
            handleDeallocQubit(builder, rDeallocQbOp, tracker);
        }
        else if (auto rGateOp = dyn_cast<qref::QuantumOperation>(op)) {
            handleGate(builder, rGateOp, tracker);
        }
        else if (auto callOp = dyn_cast<func::CallOp>(op)) {
            handleCall(builder, callOp, tracker);
        }
        else if (auto rCompbasisOp = dyn_cast<qref::ComputationalBasisOp>(op)) {
            handleCompbasis(builder, rCompbasisOp, tracker);
        }
        else if (auto rNamedObsOp = dyn_cast<qref::NamedObsOp>(op)) {
            handleNamedObs(builder, rNamedObsOp, tracker);
        }
        else if (auto rHermitianOp = dyn_cast<qref::HermitianOp>(op)) {
            handleHermitian(builder, rHermitianOp, tracker);
        }
        else if (auto rMeasureOp = dyn_cast<qref::MeasureOp>(op)) {
            handleMeasure(builder, rMeasureOp, tracker);
        }
        else if (auto rMeasureInBasisOp = dyn_cast<qref::MeasureInBasisOp>(op)) {
            handleMeasureInBasis(builder, rMeasureInBasisOp, tracker);
        }
        else if (auto adjointOp = dyn_cast<qref::AdjointOp>(op)) {
            handleAdjoint(builder, adjointOp, tracker);
        }
        else if (auto ctrlOp = dyn_cast<qref::CtrlOp>(op)) {
            handleCtrl(builder, ctrlOp, tracker);
        }
        else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
            handleIf(builder, ifOp, tracker);
        }
        else if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
            handleSwitch(builder, switchOp, tracker);
        }
        else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
            handleFor(builder, forOp, tracker);
        }
        else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
            handleWhile(builder, whileOp, tracker);
        }
        else if (auto rGraphStatePrepOp = dyn_cast<qref::GraphStatePrepOp>(op)) {
            handleGraphStatePrep(builder, rGraphStatePrepOp, tracker);
        }
    });
}

} // anonymous namespace

namespace llvm {

// Boilerplate to enable using `rQubitGetOpInfo` as DenseMap keys.
template <> struct DenseMapInfo<rQubitGetOpInfo> {
    static inline rQubitGetOpInfo getEmptyKey()
    {
        return rQubitGetOpInfo(DenseMapInfo<Value>::getEmptyKey(), -1);
    }

    static inline rQubitGetOpInfo getTombstoneKey()
    {
        return rQubitGetOpInfo(DenseMapInfo<Value>::getTombstoneKey(), -2);
    }

    static unsigned getHashValue(const rQubitGetOpInfo &val)
    {
        return hash_combine(hash_value(val.reg.getAsOpaquePointer()), val.idxAttr,
                            val.idx ? static_cast<size_t>(hash_value(val.idx.getAsOpaquePointer()))
                                    : 0);
    }

    static bool isEqual(const rQubitGetOpInfo &lhs, const rQubitGetOpInfo &rhs)
    {
        return lhs == rhs;
    }
};
} // namespace llvm

namespace catalyst {
namespace qref {

#define GEN_PASS_DECL_VALUESEMANTICSCONVERSIONPASS
#define GEN_PASS_DEF_VALUESEMANTICSCONVERSIONPASS
#include "QRef/Transforms/Passes.h.inc"

struct ValueSemanticsConversionPass
    : impl::ValueSemanticsConversionPassBase<ValueSemanticsConversionPass> {
    using ValueSemanticsConversionPassBase::ValueSemanticsConversionPassBase;

    void runOnOperation() final
    {
        Operation *mod = getOperation();
        MLIRContext *ctx = mod->getContext();
        IRRewriter builder(ctx);

        WalkResult getOpVerification = mod->walk([&](qref::GetOp getOp) {
            if (!llvm::all_of(getOp->getUsers(),
                              llvm::IsaPred<qref::QuantumOperation, qref::MeasureOp,
                                            qref::ComputationalBasisOp, qref::NamedObsOp,
                                            qref::HermitianOp, qref::MeasureInBasisOp>)) {
                getOp.emitOpError(
                    "qref.get operations can only be used by qref dialect gate operations");
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
        if (getOpVerification.wasInterrupted()) {
            return signalPassFailure();
        }

        // Collect all functions that need to be converted
        SetVector<func::FuncOp> targetFuncs;
        mod->walk([&](func::FuncOp f) {
            if (f->hasAttrOfType<UnitAttr>("quantum.node")) {
                targetFuncs.insert(f);
            }
        });

        // Convert subroutines and their corresponding calls
        // We traverse the call graph depth-first and post-order (which is guaranteed by the SCC
        // iterator), so that a caller subroutine can correctly collect the qref operands on its
        // call op to a callee subroutine.
        const CallGraph callGraph(mod);

        for (auto scc = llvm::scc_begin(&callGraph); !scc.isAtEnd(); ++scc) {
            if ((*scc->begin())->isExternal()) {
                continue;
            }

            if (!isa<func::FuncOp>((*scc->begin())->getCallableRegion()->getParentOp())) {
                continue;
            }

            func::FuncOp subroutine =
                cast<func::FuncOp>((*scc->begin())->getCallableRegion()->getParentOp());
            if (scc.hasCycle()) {
                subroutine.emitOpError("Quantum subroutine call graphs must not have cycles");
                return signalPassFailure();
            }

            if (!isQrefSubroutine(subroutine)) {
                continue;
            }

            SubroutineInfo info(subroutine);
            handleSubroutine(builder, subroutine, info.getNecessarySubroutineRValues());

            auto uses = SymbolTable::getSymbolUses(subroutine, mod);
            if (uses) {
                for (auto use : *uses) {
                    Operation *user = use.getUser();
                    if (auto callOp = dyn_cast<func::CallOp>(user)) {
                        stageCallOpForConversion(builder, callOp, info);
                    }
                }
            }
        }

        // Convert the main quantum.mode functions
        for (auto targetFunc : targetFuncs) {
            QubitValueTracker tracker;
            handleRegion(builder, targetFunc.getBody(), tracker);
            eraseAllRemainingAnchorRValues(targetFunc);
        }
    }
};

} // namespace qref
} // namespace catalyst
