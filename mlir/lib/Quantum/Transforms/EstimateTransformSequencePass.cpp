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

#include <fstream>
#include <functional>
#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "Catalyst/Analysis/ResourceAnalysis.h"
#include "Catalyst/Analysis/ResourceEffect.h"
#include "Catalyst/Analysis/ResourceResult.h"
#include "Quantum/Transforms/Passes.h"

#include "GraphPrep.hpp"
#include "ResourceProjection.hpp"

using namespace mlir;

namespace catalyst {
namespace quantum {

#define GEN_PASS_DEF_ESTIMATETRANSFORMSEQUENCEPASS
#include "Quantum/Transforms/Passes.h.inc"

namespace {

/// Remove the transform sequences from `module`, and return a copy of them with their ops, in the
/// order `apply-transform-sequence` would apply them.
FailureOr<OwningOpRef<ModuleOp>>
takeTransformSequences(ModuleOp module,
                       SmallVectorImpl<transform::ApplyRegisteredPassOp> &passOps) {
    OwningOpRef<ModuleOp> transformer;
    module.walk([&](ModuleOp op) {
        if (op->hasAttr(transform::TransformDialect::kWithNamedSequenceAttrName)) {
            transformer = op.clone();
            op.erase();
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (!transformer) {
        return transformer;
    }

    WalkResult result = transformer->walk([&](transform::NamedSequenceOp sequence) {
        for (Operation &op : sequence.getBody().front().without_terminator()) {
            auto passOp = dyn_cast<transform::ApplyRegisteredPassOp>(op);
            if (!passOp) {
                op.emitError("resource estimation only supports transform.apply_registered_pass");
                return WalkResult::interrupt();
            }
            passOps.push_back(passOp);
        }
        return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
        return failure();
    }
    return transformer;
}

/// The options of a `transform.apply_registered_pass` op as the string its pass parses, as
/// `transform.apply_registered_pass` itself builds it.
FailureOr<std::string> getPassOptions(transform::ApplyRegisteredPassOp op) {
    if (!op.getDynamicOptions().empty()) {
        op.emitError("resource estimation does not support pass options given as params");
        return failure();
    }

    std::string options;
    llvm::raw_string_ostream os(options);
    std::function<void(Attribute)> printValue = [&](Attribute value) {
        if (auto array = dyn_cast<ArrayAttr>(value)) {
            llvm::interleave(array, os, printValue, ",");
        } else if (auto str = dyn_cast<StringAttr>(value)) {
            os << str.getValue();
        } else {
            value.print(os, /*elideType=*/true);
        }
    };
    llvm::interleave(
        op.getOptions(), os,
        [&](NamedAttribute option) {
            os << option.getName().getValue() << "=";
            printValue(option.getValue());
        },
        " ");
    return options;
}

/// Add the pass of `op`, configured with its options, to `pm`, which owns it but never runs it.
FailureOr<const Pass *> createPass(transform::ApplyRegisteredPassOp op, OpPassManager &pm) {
    const PassInfo *info = PassInfo::lookup(op.getPassName());
    if (!info) {
        op.emitError() << "'" << op.getPassName() << "' is not a registered pass";
        return failure();
    }
    FailureOr<std::string> options = getPassOptions(op);
    if (failed(options)) {
        return failure();
    }
    auto onError = [&](const Twine &message) {
        op.emitError(message);
        return failure();
    };
    if (failed(info->addToPipeline(pm, *options, onError))) {
        return failure();
    }
    return &*pm.getPasses().begin();
}

/// The JSON of a level, formatted right away: it borrows names from `resources`, which the next
/// passes change.
std::string formatLevel(unsigned level, std::optional<StringRef> passName,
                        const llvm::StringMap<ResourceResult> &resources) {
    llvm::json::Object functions;
    for (const auto &entry : resources) {
        functions[entry.getKey()] = entry.getValue().toJson();
    }
    llvm::json::Object levelObject;
    levelObject["level"] = level;
    levelObject["pass"] = passName ? llvm::json::Value(*passName) : llvm::json::Value(nullptr);
    levelObject["resources"] = std::move(functions);
    return llvm::formatv("{0:2}", llvm::json::Value(std::move(levelObject))).str();
}

} // namespace

struct EstimateTransformSequencePass
    : public impl::EstimateTransformSequencePassBase<EstimateTransformSequencePass> {
    using EstimateTransformSequencePassBase::EstimateTransformSequencePassBase;

    void runOnOperation() final {
        ModuleOp module = getOperation();

        SmallVector<transform::ApplyRegisteredPassOp> passOps;
        FailureOr<OwningOpRef<ModuleOp>> transformer = takeTransformSequences(module, passOps);
        if (failed(transformer)) {
            return signalPassFailure();
        }

        // Each pass is owned by its pass manager, which must outlive the use of its options.
        SmallVector<OpPassManager> passManagers(passOps.size());
        SmallVector<const Pass *> passes;
        for (auto [op, pm] : llvm::zip(passOps, passManagers)) {
            FailureOr<const Pass *> pass = createPass(op, pm);
            if (failed(pass)) {
                return signalPassFailure();
            }
            passes.push_back(*pass);
        }

        std::optional<GraphPrep::PrepOptions> decomposition =
            passes.empty() ? std::nullopt : GraphPrep::getGraphDecompositionOptions(*passes[0]);
        ResourceAnalysis analysis(module, {},
                                  /*collectDetailedOperations=*/decomposition.has_value());

        SmallVector<std::string> levels;
        levels.push_back(formatLevel(0, std::nullopt, analysis.getResults()));

        llvm::StringMap<ResourceResult> resources;
        if (decomposition) {
            FailureOr<llvm::StringMap<ResourceResult>> projected =
                projectDecomposition(module, analysis, *decomposition);
            if (failed(projected)) {
                return signalPassFailure();
            }
            resources = std::move(*projected);
        } else {
            resources = analysis.takeResults();
        }

        for (auto [index, op, pass] : llvm::enumerate(passOps, passes)) {
            unsigned level = index + 1;
            if (level > 1 || !decomposition) {
                if (failed(applyEffect(op, *pass, resources))) {
                    return signalPassFailure();
                }
            }
            levels.push_back(formatLevel(level, op.getPassName(), resources));
        }

        writeJson("[\n" + llvm::join(levels, ",\n") + "\n]\n");
    }

  private:
    LogicalResult applyEffect(transform::ApplyRegisteredPassOp op, const Pass &pass,
                              llvm::StringMap<ResourceResult> &resources) const {
        if (GraphPrep::getGraphDecompositionOptions(pass)) {
            return op.emitError("resource estimation only supports graph-decomposition as the "
                                "first pass of the transform sequence");
        }

        const ResourceEffect *effect = ResourceEffectRegistry::get().lookup(op.getPassName());
        if (!effect) {
            if (!allowUnknownPasses) {
                return op.emitError() << "no resource effect is declared for the pass '"
                                      << op.getPassName() << "'";
            }
            op.emitWarning() << "no resource effect is declared for the pass '" << op.getPassName()
                             << "'; it is assumed not to change the resources";
            return success();
        }

        if (failed(effect->verify(pass, op.getLoc()))) {
            return failure();
        }
        for (auto &function : resources) {
            effect->apply(function.getValue(), pass);
        }
        return success();
    }

    void writeJson(const std::string &json) const {
        if (outputFname.empty()) {
            llvm::outs() << json;
            return;
        }
        std::ofstream file(outputFname);
        if (!file.is_open()) {
            llvm::errs() << "Error: could not open resource output file: " << outputFname << "\n";
            return;
        }
        file << json;
    }
};

} // namespace quantum
} // namespace catalyst
