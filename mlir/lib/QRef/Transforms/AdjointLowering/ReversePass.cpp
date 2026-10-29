// Copyright 2023 Xanadu Quantum Technologies Inc.

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

#include <algorithm>
#include <cstdint>
#include <queue>
#include <string>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Catalyst/IR/CatalystOps.h"
#include "PBC/IR/PBCOps.h"
#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"
#include "QRef/IR/QRefTypes.h"
#include "QRef/Transforms/Patterns.h"

#include "AdjointLowering.hpp"
#include "QuantumCache.hpp"

using namespace mlir;
using namespace catalyst;
using namespace catalyst::qref;

namespace {

/// Convert a row-major linearized index into per-dimension coordinates given the size of each
/// dimension, so that an arbitrary-rank tensor can be rebuilt with a single flattened loop.
SmallVector<Value> delinearizeIndex(OpBuilder &builder, Location loc, Value linear,
                                    ArrayRef<Value> dimSizes) {
    int64_t rank = dimSizes.size();
    SmallVector<Value> strides(rank);
    Value acc = index::ConstantOp::create(builder, loc, 1);
    for (int64_t d = rank - 1; d >= 0; --d) {
        strides[d] = acc;
        acc = index::MulOp::create(builder, loc, acc, dimSizes[d]);
    }
    SmallVector<Value> coords(rank);
    for (int64_t d = 0; d < rank; ++d) {
        Value q = index::DivUOp::create(builder, loc, linear, strides[d]);
        coords[d] = index::RemUOp::create(builder, loc, q, dimSizes[d]);
    }
    return coords;
}

/// Emit the adjoint gates and reversed control flow from the recorded classical execution.
class AdjointGenerator {
  public:
    AdjointGenerator(IRMapping &remappedValues, QuantumCache &cache, Region &adjointRegion)
        : remappedValues(remappedValues), cache(cache), adjointRegion(adjointRegion) {}

    LogicalResult generate(Region &region, OpBuilder &builder) {
        generateImpl(region, builder);
        return failure(generationFailed);
    }

  private:
    void generateImpl(Region &region, OpBuilder &builder) {
        assert(region.hasOneBlock() &&
               "Expected only structured control flow (each region should have a single block)");
        if (generationFailed) {
            return;
        }
        for (Operation &op : llvm::reverse(region.front())) {
            if (op.hasTrait<OpTrait::IsTerminator>()) {
                continue;
            }
            if (auto callOp = dyn_cast<func::CallOp>(op)) {
                visitOperation(callOp, builder);
            } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
                visitOperation(forOp, builder);
            } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
                visitOperation(ifOp, builder);
            } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
                visitOperation(whileOp, builder);
            } else if (auto switchOp = dyn_cast<scf::IndexSwitchOp>(op)) {
                visitOperation(switchOp, builder);
            } else if (isa<GetOp>(op)) {
                // Recreated at each reversed use, once its register has been allocated.
            } else if (auto allocOp = dyn_cast<AllocOp>(op)) {
                DeallocOp::create(builder, allocOp.getLoc(),
                                  remappedValues.lookupOrDefault(allocOp.getQreg()));
            } else if (auto deallocOp = dyn_cast<DeallocOp>(op)) {
                auto sourceAlloc = deallocOp.getQreg().getDefiningOp<AllocOp>();
                if (!sourceAlloc) {
                    deallocOp.emitError("Unable to reverse dynamic register deallocation in the "
                                        "adjoint region: allocation size could not be determined");
                    generationFailed = true;
                    return;
                }
                Value nqubits = sourceAlloc.getNqubits();
                if (nqubits) {
                    nqubits =
                        isAvailableToReversePass(nqubits, adjointRegion)
                            ? remappedValues.lookupOrDefault(nqubits)
                            : ListPopOp::create(builder, deallocOp.getLoc(), cache.wireVector);
                }
                auto newAlloc =
                    AllocOp::create(builder, deallocOp.getLoc(), deallocOp.getQreg().getType(),
                                    nqubits, sourceAlloc.getNqubitsAttrAttr());
                remappedValues.map(deallocOp.getQreg(), newAlloc.getQreg());
            } else if (auto allocOp = dyn_cast<AllocQubitOp>(op)) {
                DeallocQubitOp::create(builder, allocOp.getLoc(),
                                       remappedValues.lookupOrDefault(allocOp.getQubit()));
            } else if (auto deallocOp = dyn_cast<DeallocQubitOp>(op)) {
                auto newAlloc = AllocQubitOp::create(builder, deallocOp.getLoc());
                remappedValues.map(deallocOp.getQubit(), newAlloc.getQubit());
            } else if (auto gate = dyn_cast<qref::QuantumGate>(op)) {
                remapReferences(op.getOperands(), builder);
                visitOperation(gate, builder);
            } else if (auto ppr = dyn_cast<pbc::RefPPRotationOp>(op)) {
                remapReferences(op.getOperands(), builder);
                auto clone = cast<pbc::RefPPRotationOp>(builder.clone(op, remappedValues));
                clone.setRotationKind(-ppr.getRotationKind());
            } else if (hasQuantumOperations(&op)) {
                op.emitError("Unhandled operation in adjoint region");
                generationFailed = true;
            }
            if (generationFailed) {
                return;
            }
        }
    }

    // Match the forward pass's per-use wire recording in reverse operand order.
    void remapReferences(ValueRange operands, OpBuilder &builder) {
        for (Value operand : llvm::reverse(operands)) {
            auto get = operand.getDefiningOp<GetOp>();
            if (!get || !adjointRegion.isAncestor(get->getParentRegion())) {
                continue;
            }
            Value index = get.getIdx();
            if (index) {
                index = isAvailableToReversePass(index, adjointRegion)
                            ? remappedValues.lookupOrDefault(index)
                            : ListPopOp::create(builder, get.getLoc(), cache.wireVector);
            }
            auto clone = GetOp::create(builder, get.getLoc(), get.getQubit().getType(),
                                       remappedValues.lookupOrDefault(get.getQreg()), index,
                                       get.getIdxAttrAttr());
            remappedValues.map(operand, clone.getQubit());
        }
    }

    void visitOperation(qref::QuantumGate gate, OpBuilder &builder) {
        auto clone = cast<QuantumGate>(builder.clone(*gate, remappedValues));
        clone.setAdjointFlag(!gate.getAdjointFlag());
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(clone);
        auto controlValues = llvm::to_vector(gate.getCtrlValueOperands());
        for (Value &controlValue : llvm::reverse(controlValues)) {
            controlValue =
                isAvailableToReversePass(controlValue, adjointRegion)
                    ? remappedValues.lookupOrDefault(controlValue)
                    : ListPopOp::create(builder, gate.getLoc(), cache.controlValueVector);
        }
        clone.setCtrlValueOperands(controlValues);
        if (auto parametrizedGate = dyn_cast<ParametrizedGate>(gate.getOperation())) {
            auto params = recoverParameters(gate, parametrizedGate.getAllParams(), builder);
            if (generationFailed) {
                return;
            }
            MutableOperandRange(clone, parametrizedGate.getParamOperandIdx(), params.size())
                .assign(params);
        }
    }

    SmallVector<Value> recoverParameters(Operation *operation, ValueRange params,
                                         OpBuilder &builder) {
        size_t numParams = params.size();
        SmallVector<Value> cachedParams(numParams);
        size_t idx = 0;
        // popping gives the parameters in reverse
        for (Value param : llvm::reverse(params)) {
            Type paramType = param.getType();

            // The forward pass does not record params that are already available here; reuse
            // the value directly instead of popping.
            if (isAvailableToReversePass(param, adjointRegion)) {
                cachedParams[numParams - 1 - idx] = remappedValues.lookupOrDefault(param);
                idx++;
                continue;
            }

            if (mlir::failed(verifyTypeIsCacheable(paramType, operation))) {
                generationFailed = true;
                return {};
            }
            if (paramType.isF64()) {
                cachedParams[numParams - 1 - idx] =
                    ListPopOp::create(builder, operation->getLoc(), cache.paramVector);
                idx++;
                continue;
            }

            // This is a temporary trick to make it lossless for widths <= 64 bits.
            Type i64Ty = builder.getI64Type();
            auto truncFromI64 = [&](Value v, Type target, Location l) -> Value {
                if (target == i64Ty) {
                    return v;
                }
                return arith::TruncIOp::create(builder, l, target, v);
            };
            if (auto intType = dyn_cast<IntegerType>(paramType)) {
                Location l = operation->getLoc();
                Value popped = ListPopOp::create(builder, l, cache.intVector);
                cachedParams[numParams - 1 - idx] = truncFromI64(popped, intType, l);
                idx++;
                continue;
            }

            // Guaranteed by verifyTypeIsCacheable above.
            auto aTensorType = cast<RankedTensorType>(paramType);
            ArrayRef<int64_t> shape = aTensorType.getShape();
            Type elementType = aTensorType.getElementType();
            // Constants
            auto loc = operation->getLoc();

            // Real-valued tensor params (e.g. `qref.operator` angle tensors, or a
            // BasisRotation's `tensor<NxNxf64>` matrix) were cached element-by-element in
            // row-major order in `paramVector`; rebuild the tensor by popping them back.
            // Integer/boolean tensors (e.g. a MultiX `tensor<Nxi1>` bitstring or a QROM
            // `tensor<Nxi64>` bitstring) were recorded the same way in the i64 buffer
            // `intVector` (widths > 64 bits already rejected by verifyTypeIsCacheable);
            // truncate each popped element back on rebuild.
            // Note that the complex-matrix path below is specific to QubitUnitary. We can
            // revisit this when we have more complex-valued tensor params to handle.
            if (elementType.isF64() || elementType.isInteger()) {
                // Pop one element from the buffer matching the tensor's element type.
                auto readElement = [&]() -> Value {
                    if (elementType.isInteger()) {
                        Value popped = ListPopOp::create(builder, loc, cache.intVector);
                        return truncFromI64(popped, elementType, loc);
                    }
                    return ListPopOp::create(builder, loc, cache.paramVector);
                };
                if (aTensorType.getRank() == 0) {
                    Value element = readElement();
                    cachedParams[numParams - 1 - idx] =
                        tensor::FromElementsOp::create(builder, loc, paramType, element);
                    idx++;
                    continue;
                }
                Value c0i = index::ConstantOp::create(builder, loc, 0);
                Value c1i = index::ConstantOp::create(builder, loc, 1);
                SmallVector<Value> dimSizes;
                SmallVector<Value> dynSizes;
                Value total = c1i;
                for (int64_t d = 0; d < aTensorType.getRank(); ++d) {
                    Value dim;
                    if (ShapedType::kDynamic != shape[d]) {
                        dim = index::ConstantOp::create(builder, loc, shape[d]);
                    } else {
                        dim = tensor::DimOp::create(builder, loc, param, d);
                        dynSizes.push_back(dim);
                    }
                    dimSizes.push_back(dim);
                    total = index::MulOp::create(builder, loc, total, dim);
                }
                Value empty = tensor::EmptyOp::create(builder, loc, aTensorType, dynSizes);
                // Elements were pushed in ascending linearized order, so popping yields them in
                // descending order: place the i-th popped value at linearized index
                // (total - 1 - i), delinearized back into per-dimension coordinates.
                scf::ForOp loop =
                    scf::ForOp::create(builder, loc, c0i, total, c1i, ValueRange{empty});
                {
                    OpBuilder::InsertionGuard guard(builder);
                    builder.setInsertionPointToStart(loop.getBody());
                    Value acc = loop.getRegionIterArg(0);
                    Value iPlusOne =
                        index::AddOp::create(builder, loc, loop.getInductionVar(), c1i);
                    Value revLinear = index::SubOp::create(builder, loc, total, iPlusOne);
                    SmallVector<Value> coords = delinearizeIndex(builder, loc, revLinear, dimSizes);
                    Value element = readElement();
                    Value updated = tensor::InsertOp::create(builder, loc, element, acc, coords);
                    scf::YieldOp::create(builder, loc, updated);
                }
                cachedParams[numParams - 1 - idx] = loop.getResult(0);
                idx++;
                continue;
            }

            Value c0 = index::ConstantOp::create(builder, loc, 0);
            Value c1 = index::ConstantOp::create(builder, loc, 1);
            // TODO: Generalize to all possible dimensions
            bool isDim0Static = ShapedType::kDynamic != shape[0];
            bool isDim1Static = ShapedType::kDynamic != shape[1];
            Value dim0Length = isDim0Static
                                   ? (Value)index::ConstantOp::create(builder, loc, shape[0])
                                   : (Value)tensor::DimOp::create(builder, loc, param, c0);
            Value dim1Length = isDim1Static
                                   ? (Value)index::ConstantOp::create(builder, loc, shape[1])
                                   : (Value)tensor::DimOp::create(builder, loc, param, c1);

            // Renaming for legibility
            // Note: Since this is a square matrix, upperBound for both loops is the
            // same value.
            Value lowerBoundDim0 = c0;
            Value upperBoundDim0 = dim0Length;
            Value stepDim0 = c1;
            Value lowerBoundDim1 = c0;
            Value upperBoundDim1 = dim1Length;
            Value stepDim1 = c1;
            Value beginningTensor = tensor::EmptyOp::create(builder, loc, shape, elementType);
            // This time, we are in reverse, so we need to start
            // with N-1 since MLIR does not allow for loops with negative step sizes.
            SmallVector<Value> initialValues = {beginningTensor};

            scf::ForOp iForLoop = scf::ForOp::create(builder, loc, lowerBoundDim0, upperBoundDim0,
                                                     stepDim0, initialValues);
            {
                OpBuilder::InsertionGuard afterIForLoop(builder);
                builder.setInsertionPointToStart(iForLoop.getBody());
                auto iIterArgs = iForLoop.getRegionIterArgs();
                Value currIthTensor = iIterArgs.front();

                Value i = iForLoop.getInductionVar();
                Value iPlusOne = index::AddOp::create(builder, loc, i, c1);
                Value nMinusIMinusOne = index::SubOp::create(builder, loc, dim0Length, iPlusOne);
                // Just for legibility
                Value iTensorIndex = nMinusIMinusOne;

                scf::ForOp jForLoop = scf::ForOp::create(builder, loc, lowerBoundDim1,
                                                         upperBoundDim1, stepDim1, currIthTensor);
                {
                    OpBuilder::InsertionGuard afterJForLoop(builder);
                    builder.setInsertionPointToStart(jForLoop.getBody());
                    auto jIterArgs = jForLoop.getRegionIterArgs();
                    assert(jIterArgs.size() == 1 &&
                           "jForLoop has more induction variables than necessary.");
                    Value currIthJthTensor = jIterArgs.front();

                    Value imag = ListPopOp::create(builder, loc, cache.paramVector);
                    Value real = ListPopOp::create(builder, loc, cache.paramVector);
                    Value element =
                        complex::CreateOp::create(builder, loc, elementType, real, imag);

                    // TODO: Generalize to types which are not complex
                    Value j = jForLoop.getInductionVar();
                    Value jPlusOne = index::AddOp::create(builder, loc, j, c1);
                    Value nMinusJMinusOne =
                        index::SubOp::create(builder, loc, dim1Length, jPlusOne);
                    // Just for legibility
                    Value jTensorIndex = nMinusJMinusOne;
                    SmallVector<Value> indices = {iTensorIndex, jTensorIndex};

                    Value updatedIthJthTensor =
                        tensor::InsertOp::create(builder, loc, element, currIthJthTensor, indices);
                    scf::YieldOp::create(builder, loc, updatedIthJthTensor);
                }

                Value ithTensor = jForLoop.getResult(0);
                scf::YieldOp::create(builder, loc, ithTensor);
            }

            Value recreatedTensor = iForLoop.getResult(0);
            cachedParams[numParams - 1 - idx] = recreatedTensor;
            idx++;
        }
        return cachedParams;
    }

    void visitOperation(func::CallOp callOp, OpBuilder &builder) {
        if (!hasQuantumOperations(callOp)) {
            return;
        }
        auto funcOp =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(callOp, callOp.getCalleeAttr());
        if (!funcOp || funcOp.isExternal() || !funcOp.getFunctionType().getResults().empty()) {
            callOp.emitError(
                "Adjoint lowering requires a quantum callee with a body and no results");
            generationFailed = true;
            return;
        }

        remapReferences(callOp.getOperands(), builder);
        SmallVector<Value> classicalArgs;
        for (Value arg : callOp.getOperands()) {
            if (!isa<QuregType, QubitType>(arg.getType())) {
                classicalArgs.push_back(arg);
            }
        }
        auto params = recoverParameters(callOp, classicalArgs, builder);
        if (generationFailed) {
            return;
        }
        SmallVector<Value> args;
        unsigned index = 0;
        for (Value arg : callOp.getOperands()) {
            args.push_back(isa<QuregType, QubitType>(arg.getType())
                               ? remappedValues.lookupOrDefault(arg)
                               : params[index++]);
        }
        auto adjointFn = getOrCreateAdjointFunction(funcOp, builder);
        func::CallOp::create(builder, callOp.getLoc(), adjointFn, args);
    }

    /// Reuse one private adjoint function per callee. References keep the original signature.
    func::FuncOp getOrCreateAdjointFunction(func::FuncOp funcOp, OpBuilder &builder) {
        ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
        std::string adjointName = funcOp.getName().str() + ".adjoint";
        if (auto existing = moduleOp.lookupSymbol<func::FuncOp>(adjointName)) {
            return existing;
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(moduleOp.getBody());
        auto adjointFn =
            func::FuncOp::create(builder, funcOp.getLoc(), adjointName, funcOp.getFunctionType());
        adjointFn.setPrivate();
        auto *entry = adjointFn.addEntryBlock();
        builder.setInsertionPointToStart(entry);
        auto adjoint = AdjointOp::create(builder, funcOp.getLoc());
        auto *body = new Block();
        adjoint.getRegion().push_back(body);
        builder.setInsertionPointToStart(body);
        IRMapping mapping;
        mapping.map(funcOp.getArguments(), entry->getArguments());
        for (Operation &op : funcOp.front().without_terminator()) {
            builder.clone(op, mapping);
        }
        builder.setInsertionPointAfter(adjoint);
        func::ReturnOp::create(builder, funcOp.getLoc());
        return adjointFn;
    }

    void visitOperation(scf::ForOp forOp, OpBuilder &builder) {
        if (!hasQuantumOperations(forOp)) {
            return;
        }
        Value tape = cache.controlFlowTapes.at(forOp);
        auto recoverBound = [&](Value original) -> Value {
            if (std::optional<int64_t> constant = getConstantIntValue(original)) {
                return index::ConstantOp::create(builder, forOp.getLoc(), *constant);
            }
            return ListPopOp::create(builder, forOp.getLoc(), tape);
        };
        Value step = recoverBound(forOp.getStep());
        Value stop = recoverBound(forOp.getUpperBound());
        Value start = recoverBound(forOp.getLowerBound());
        auto reversed =
            scf::ForOp::create(builder, forOp.getLoc(), start, stop, step, ValueRange{},
                               [&](OpBuilder &bodyBuilder, Location loc, Value, ValueRange) {
                                   generateImpl(forOp.getRegion(), bodyBuilder);
                                   scf::YieldOp::create(bodyBuilder, loc);
                               });
        reversed->setDiscardableAttrs(forOp->getDiscardableAttrDictionary());
    }

    void visitOperation(scf::IfOp ifOp, OpBuilder &builder) {
        if (!hasQuantumOperations(ifOp)) {
            return;
        }
        Value condition =
            ListPopOp::create(builder, ifOp.getLoc(), cache.controlFlowTapes.at(ifOp));
        condition = index::CastSOp::create(builder, ifOp.getLoc(), builder.getI1Type(), condition);
        auto fill = [&](Region &region) {
            return [&](OpBuilder &bodyBuilder, Location loc) {
                if (!region.empty()) {
                    generateImpl(region, bodyBuilder);
                }
                scf::YieldOp::create(bodyBuilder, loc);
            };
        };
        auto reversed = scf::IfOp::create(builder, ifOp.getLoc(), condition,
                                          fill(ifOp.getThenRegion()), fill(ifOp.getElseRegion()));
        reversed->setDiscardableAttrs(ifOp->getDiscardableAttrDictionary());
    }

    void visitOperation(scf::WhileOp whileOp, OpBuilder &builder) {
        if (!hasQuantumOperations(whileOp)) {
            return;
        }
        Value count =
            ListPopOp::create(builder, whileOp.getLoc(), cache.controlFlowTapes.at(whileOp));
        // Undo the final condition evaluation, then each after/before pair in reverse order.
        generateImpl(whileOp.getBefore(), builder);
        Value zero = index::ConstantOp::create(builder, whileOp.getLoc(), 0);
        Value one = index::ConstantOp::create(builder, whileOp.getLoc(), 1);
        auto reversed =
            scf::ForOp::create(builder, whileOp.getLoc(), zero, count, one, ValueRange{},
                               [&](OpBuilder &bodyBuilder, Location loc, Value, ValueRange) {
                                   generateImpl(whileOp.getAfter(), bodyBuilder);
                                   generateImpl(whileOp.getBefore(), bodyBuilder);
                                   scf::YieldOp::create(bodyBuilder, loc);
                               });
        reversed->setDiscardableAttrs(whileOp->getDiscardableAttrDictionary());
    }

    void visitOperation(scf::IndexSwitchOp switchOp, OpBuilder &builder) {
        if (!hasQuantumOperations(switchOp)) {
            return;
        }
        Value index =
            ListPopOp::create(builder, switchOp.getLoc(), cache.controlFlowTapes.at(switchOp));
        auto reversed = scf::IndexSwitchOp::create(builder, switchOp.getLoc(), TypeRange{}, index,
                                                   switchOp.getCases(), switchOp.getNumCases());
        reversed->setDiscardableAttrs(switchOp->getDiscardableAttrDictionary());
        for (auto [oldRegion, newRegion] :
             llvm::zip_equal(switchOp->getRegions(), reversed->getRegions())) {
            OpBuilder::InsertionGuard guard(builder);
            newRegion.push_back(new Block());
            builder.setInsertionPointToStart(&newRegion.front());
            generateImpl(oldRegion, builder);
            scf::YieldOp::create(builder, switchOp.getLoc());
        }
    }

    IRMapping &remappedValues;
    QuantumCache &cache;
    Region &adjointRegion;
    bool generationFailed = false;
};

} // namespace

namespace catalyst::qref {

LogicalResult generateAdjointReversePass(Region &region, OpBuilder &builder,
                                         IRMapping &remappedValues, QuantumCache &cache) {
    AdjointGenerator generator{remappedValues, cache, region};
    return generator.generate(region, builder);
}

} // namespace catalyst::qref
