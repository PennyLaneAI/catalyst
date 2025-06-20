// Copyright 2024-2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Ion/IR/IonOps.h"
#include "Ion/Transforms/Passes.h"
#include "Ion/Transforms/Patterns.h"
#include "Ion/Transforms/oqd_database_managers.hpp"
#include "Ion/Transforms/oqd_database_types.hpp"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;

namespace catalyst {
namespace ion {

#define GEN_PASS_DECL_GATESTOPULSESPASS
#define GEN_PASS_DEF_GATESTOPULSESPASS
#include "Ion/Transforms/Passes.h.inc"

struct GatesToPulsesPass : impl::GatesToPulsesPassBase<GatesToPulsesPass> {
    using GatesToPulsesPassBase::GatesToPulsesPassBase;

    LevelAttr getLevelAttr(MLIRContext *ctx, IRRewriter &builder, Level level)
    {
        return LevelAttr::get(
            ctx, builder.getStringAttr(level.label), builder.getI64IntegerAttr(level.principal),
            builder.getF64FloatAttr(level.spin), builder.getF64FloatAttr(level.orbital),
            builder.getF64FloatAttr(level.nuclear), builder.getF64FloatAttr(level.spin_orbital),
            builder.getF64FloatAttr(level.spin_orbital_nuclear),
            builder.getF64FloatAttr(level.spin_orbital_nuclear_magnetization),
            builder.getF64FloatAttr(level.energy));
    }

    TransitionAttr getTransitionAttr(MLIRContext *ctx, IRRewriter &builder, Transition transition)
    {
        return TransitionAttr::get(ctx, builder.getStringAttr(transition.level_0),
                                   builder.getStringAttr(transition.level_1),
                                   builder.getF64FloatAttr(transition.einstein_a),
                                   builder.getStringAttr(transition.multipole));
    }

    PhononAttr getPhononAttr(MLIRContext *ctx, IRRewriter &builder, Phonon phonon)
    {
        return PhononAttr::get(ctx, builder.getF64FloatAttr(phonon.energy),
                               builder.getDenseF64ArrayAttr(phonon.eigenvector));
    }

    bool canScheduleOn(RegisteredOperationName opInfo) const override
    {
        return opInfo.hasInterface<FunctionOpInterface>();
    }

    void runOnOperation() final
    {
        func::FuncOp op = cast<func::FuncOp>(getOperation());
        if (!op->hasAttr("qnode")) {
            return;
        }
        auto &context = getContext();
        ConversionTarget target(context);

        // Only quantum gate ops (CustomOp) must be eliminated; other quantum
        // ops are allowed to remain.
        target.addIllegalOp<catalyst::quantum::CustomOp>();
        target.addLegalDialect<IonDialect>();
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

        auto allocOp = *op.getOps<quantum::AllocOp>().begin();
        auto nQubits = allocOp.getNqubitsAttr().value();

        OQDDatabaseManager dataManager(DeviceTomlLoc, QubitTomlLoc, Gate2PulseDecompTomlLoc,
                                       nQubits);

        if (LoadIon) {
            // FIXME(?): we only load Yb171 ion since the hardware ion species is unlikely to change
            MLIRContext *ctx = op->getContext();
            IRRewriter builder(ctx);
            Ion ion = dataManager.getIonParams().at("Yb171");

            SmallVector<Attribute> levels, transitions;
            for (const Level &level : ion.levels) {
                levels.push_back(cast<Attribute>(getLevelAttr(ctx, builder, level)));
            }
            for (const Transition &transition : ion.transitions) {
                transitions.push_back(cast<Attribute>(getTransitionAttr(ctx, builder, transition)));
            }

            builder.setInsertionPointToStart(&(op->getRegion(0).front()));
            builder.create<ion::IonOp>(
                op->getLoc(), IonType::get(ctx), builder.getStringAttr(ion.name),
                builder.getF64FloatAttr(ion.mass), builder.getF64FloatAttr(ion.charge),
                ion.position, builder.getArrayAttr(levels), builder.getArrayAttr(transitions));

            SmallVector<Attribute> phonons;
            for (const Phonon &phonon : dataManager.getPhononParams()) {
                phonons.push_back(cast<Attribute>(getPhononAttr(ctx, builder, phonon)));
            }
            // TODO: For now, we only print one phonon to be consistent with TriCal examples,
            // but we should print all of them eventually
            builder.create<ion::ModesOp>(op->getLoc(), builder.getArrayAttr(phonons[0]));
        }

        RewritePatternSet ionPatterns(&getContext());
        populateGatesToPulsesPatterns(ionPatterns, dataManager);

        if (failed(applyPartialConversion(op, target, std::move(ionPatterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace ion

std::unique_ptr<Pass> createGatesToPulsesPass()
{
    return std::make_unique<ion::GatesToPulsesPass>();
}

} // namespace catalyst
