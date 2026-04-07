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

#include <variant>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

#include "QRef/IR/QRefOps.h"
#include "QRef/Transforms/value_semantics_conversion.h"

using namespace mlir;
using namespace catalyst;

namespace ReferenceToValueSemanticsConversion {

bool isQrefSubroutine(func::FuncOp f)
{
    // If has a qref argument, definitely is a qref subroutine
    if (llvm::any_of(f.getArgumentTypes(), llvm::IsaPred<qref::QubitType, qref::QuregType>)) {
        return true;
    }

    // If we don't know from the args, must look at the body
    if (f.isDeclaration()) {
        return false;
    }

    MLIRContext *ctx = f->getContext();
    auto *qrefDialect = ctx->getLoadedDialect<qref::QRefDialect>();

    WalkResult walkResult = f.walk([qrefDialect](Operation *op) {
        if (op->getDialect() == qrefDialect) {
            return WalkResult::interrupt();
        }
        if (func::CallOp callOp = dyn_cast<func::CallOp>(op)) {
            auto funcOp = mlir::SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
                callOp, callOp.getCalleeAttr());
            if (isQrefSubroutine(funcOp)) {
                return WalkResult::interrupt();
            }
        }
        return WalkResult::advance();
    });
    return walkResult.wasInterrupted();
}

void _getNecessarySubroutineRValues(func::FuncOp f, SetVector<Value> &necessarySubroutineRValues)
{
    auto *qrefDialect = f.getContext()->getLoadedDialect<qref::QRefDialect>();
    llvm::SmallDenseSet<Value, 8> rQregsTakenIn;

    f.walk([&](Operation *op) {
        if (op->getDialect() != qrefDialect && !isa<func::CallOp>(op)) {
            return;
        }
        if (isa<qref::GetOp>(op)) {
            return;
        }
        for (Value v : op->getOperands()) {
            if (isa<qref::QuregType>(v.getType())) {
                // Ignore allocations from inside the region itself
                if (llvm::is_contained(f.getArguments(), v)) {
                    necessarySubroutineRValues.insert(v);
                    rQregsTakenIn.insert(v);
                }
            }
            else if (isa<qref::QubitType>(v.getType())) {
                if (isa<BlockArgument>(v) || !isa<qref::GetOp>(v.getDefiningOp())) {
                    // Ignore allocations from inside the region itself
                    if (llvm::is_contained(f.getArguments(), v)) {
                        necessarySubroutineRValues.insert(v);
                    }
                }
                else {
                    Value rQreg = getRSourceRegisterValue(v);
                    if (llvm::is_contained(f.getArguments(), rQreg)) {
                        auto getOp = cast<qref::GetOp>(v.getDefiningOp());
                        if (getOp.getIdx()) {
                            // dynamic extract index, must take in the reg
                            necessarySubroutineRValues.insert(rQreg);
                            rQregsTakenIn.insert(rQreg);
                        }
                        else {
                            necessarySubroutineRValues.insert(v);
                        }
                    }
                }
            }
        }
    });

    // If any rQregs are taken in, any rQubits belonging to them must not be taken in separately
    necessarySubroutineRValues.remove_if([&](const Value &v) {
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
    necessarySubroutineRValues.remove_if([&](const Value &v) {
        if (isa<BlockArgument>(v) || !isa<qref::GetOp>(v.getDefiningOp())) {
            return false;
        }

        rQubitGetOpInfo info = getGetOpInfo(v).value();
        // If already exists in set, insertion will fail, and we have seen an alias, so need to
        // remove
        return !seenGetInfos.insert(info).second;
    });
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
        _getNecessarySubroutineRValues(this->subroutine, this->necessarySubroutineRValues);
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
            std::pair _pair = std::get<std::pair<unsigned, uint64_t>>(newArgsInfo);
            unsigned oldCallArgIdx = _pair.first;
            unsigned extractIdx = _pair.second;
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

} // namespace ReferenceToValueSemanticsConversion
