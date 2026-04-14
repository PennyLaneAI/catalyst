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

#pragma once

#include <cstdint>
#include <functional>
#include <optional>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"

#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"

using namespace mlir;
using namespace catalyst;

namespace ReferenceToValueSemanticsConversion {

// Structs holding the core conversion logic
struct QubitValueTracker;
struct TransientQubitExtractor;

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

// Misc helper functions
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

namespace {
void _getNecessaryRegionRValuesImpl(Region &r, SetVector<Value> &necessaryRegionRValues,
                                    std::function<bool(Region &, Value)> isFromOutside)
{
    auto *qrefDialect = r.getContext()->getLoadedDialect<qref::QRefDialect>();
    llvm::SmallDenseSet<Value, 8> rQregsTakenIn;

    r.walk([&](Operation *op) {
        if (op->getDialect() != qrefDialect && !isa<func::CallOp>(op)) {
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
                if (isa<BlockArgument>(v) || !isa<qref::GetOp>(v.getDefiningOp())) {
                    // Ignore allocations from inside the region itself
                    if (isFromOutside(r, v)) {
                        necessaryRegionRValues.insert(v);
                    }
                }
                else {
                    Value rQreg = getRSourceRegisterValue(v);
                    if (isFromOutside(r, rQreg)) {
                        auto getOp = cast<qref::GetOp>(v.getDefiningOp());
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
        if (isa<BlockArgument>(v) || !isa<qref::GetOp>(v.getDefiningOp())) {
            return false;
        }

        rQubitGetOpInfo info = getGetOpInfo(v).value();
        // If already exists in set, insertion will fail, and we have seen an alias, so need to
        // remove
        return !seenGetInfos.insert(info).second;
    });
}
} // namespace

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

// The main converter function
template <typename OpTy>
OpTy migrateOpToValueSemantics(IRRewriter &builder, Operation *qrefOp, QubitValueTracker &tracker,
                               std::optional<TypeRange> newResultTypes = std::nullopt);

// Individual handlers for each op
void handleAlloc(IRRewriter &builder, qref::AllocOp rAllocOp, QubitValueTracker &tracker);
void handleDealloc(IRRewriter &builder, qref::DeallocOp rDeallocOp, QubitValueTracker &tracker);
void handleAllocQubit(IRRewriter &builder, qref::AllocQubitOp rAllocQbOp,
                      QubitValueTracker &tracker);
void handleDeallocQubit(IRRewriter &builder, qref::DeallocQubitOp rDeallocQbOp,
                        QubitValueTracker &tracker);
void handleGate(IRRewriter &builder, qref::QuantumOperation rGateOp, QubitValueTracker &tracker);
void handleMeasure(IRRewriter &builder, qref::MeasureOp rMeasureOp, QubitValueTracker &tracker);
void handleCall(IRRewriter &builder, func::CallOp callOp, QubitValueTracker &tracker);
void handleCompbasis(IRRewriter &builder, qref::ComputationalBasisOp rCompbasisOp,
                     QubitValueTracker &tracker);
void handleNamedObs(IRRewriter &builder, qref::NamedObsOp rNamedObsOp, QubitValueTracker &tracker);
void handleHermitian(IRRewriter &builder, qref::HermitianOp rHermitianOp,
                     QubitValueTracker &tracker);
void handleAdjoint(IRRewriter &builder, qref::AdjointOp rAdjointOp, QubitValueTracker &tracker);
void handleIf(IRRewriter &builder, scf::IfOp ifOp, QubitValueTracker &tracker);
void handleSwitch(IRRewriter &builder, scf::IndexSwitchOp switchOp, QubitValueTracker &tracker);
void handleFor(IRRewriter &builder, scf::ForOp forOp, QubitValueTracker &tracker);
void handleWhile(IRRewriter &builder, scf::WhileOp whileOp, QubitValueTracker &tracker);
void handleSubroutine(IRRewriter &builder, func::FuncOp f,
                      const SetVector<Value> &rValuesUsedBySubroutine);

// Main driver
void handleRegion(IRRewriter &builder, Region &r, QubitValueTracker &tracker);
} // namespace ReferenceToValueSemanticsConversion

namespace llvm {

// Boilerplate to enable using `rQubitGetOpInfo` as DenseMap keys.
template <> struct DenseMapInfo<ReferenceToValueSemanticsConversion::rQubitGetOpInfo> {
    using rQubitGetOpInfo = ReferenceToValueSemanticsConversion::rQubitGetOpInfo;

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
