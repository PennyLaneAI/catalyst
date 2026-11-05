// Copyright 2026 Xanadu Quantum Technologies Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "PBC/Utils/PBCSynthesisEvaluation.h"

#include <bit>
#include <cmath>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DynamicLibrary.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;

namespace catalyst::pbc {
namespace {

constexpr size_t maxGates = 100000;

bool isRuntimeCall(func::CallOp call) {
    if (!call) {
        return false;
    }
    StringRef name = call.getCallee();
    bool writer = name == "rs_decomposition_get_gates";
    if (!writer && name != "rs_decomposition_get_size" && name != "rs_decomposition_get_phase") {
        return false;
    }
    auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(call, call.getCalleeAttr());
    if (!function || !function.isDeclaration()) {
        return false;
    }
    auto inputs = call.getOperandTypes();
    if (inputs.size() != (writer ? 4 : 3) || !inputs[writer ? 1 : 0].isF64() ||
        !inputs[writer ? 2 : 1].isF64() || !inputs[writer ? 3 : 2].isInteger(1)) {
        return false;
    }
    if (writer) {
        auto buffer = dyn_cast<MemRefType>(inputs.front());
        return call.getNumResults() == 0 && buffer && buffer.getRank() == 1 &&
               buffer.getElementType().isIndex() && buffer.getLayout().isIdentity();
    }
    return call.getNumResults() == 1 &&
           (name == "rs_decomposition_get_size" ? call.getResult(0).getType().isIndex()
                                                : call.getResult(0).getType().isF64());
}

// Restrict reads to the buffer shape emitted by gridsynth: one allocation, one
// fill, direct loads after the fill, and deallocation after all reads. Reject
// aliases, stores, escaping buffers and conditional writes.
func::CallOp immutableFill(Value buffer) {
    auto alloc = buffer.getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc.getType().getRank() != 1 || !alloc.getType().getLayout().isIdentity()) {
        return {};
    }
    func::CallOp fill;
    llvm::SmallVector<Operation *> reads;
    llvm::SmallVector<Operation *> releases;
    for (Operation *user : buffer.getUsers()) {
        if (auto call = dyn_cast<func::CallOp>(user);
            isRuntimeCall(call) && call.getCallee() == "rs_decomposition_get_gates" &&
            call.getOperand(0) == buffer) {
            if (fill || call->getBlock() != alloc->getBlock()) {
                return {};
            }
            fill = call;
        } else if (auto load = dyn_cast<memref::LoadOp>(user); load && load.getMemref() == buffer) {
            Operation *ancestor = alloc->getBlock()->findAncestorOpInBlock(*load);
            if (!ancestor) {
                return {};
            }
            reads.push_back(ancestor);
        } else if (auto release = dyn_cast<memref::DeallocOp>(user);
                   release && release->getBlock() == alloc->getBlock()) {
            releases.push_back(release);
        } else {
            return {};
        }
    }
    if (!fill || !alloc->isBeforeInBlock(fill)) {
        return {};
    }
    for (Operation *read : reads) {
        if (!fill->isBeforeInBlock(read) || llvm::any_of(releases, [&](Operation *release) {
                return !read->isBeforeInBlock(release);
            })) {
            return {};
        }
    }
    return fill;
}

struct Sequence {
    std::vector<size_t> gates;
    double phase;
};

std::shared_ptr<const Sequence> evaluateSequence(ValueRange arguments,
                                                 llvm::function_ref<Attribute(Value)> resolve) {
    auto theta = dyn_cast_or_null<FloatAttr>(resolve(arguments[0]));
    auto epsilon = dyn_cast_or_null<FloatAttr>(resolve(arguments[1]));
    auto ppr = dyn_cast_or_null<IntegerAttr>(resolve(arguments[2]));
    if (!theta || !epsilon || !ppr || !ppr.getValue().isOne() ||
        !std::isfinite(theta.getValueAsDouble()) || !std::isfinite(epsilon.getValueAsDouble()) ||
        epsilon.getValueAsDouble() <= 0) {
        return {};
    }
    auto getter =
        reinterpret_cast<void *(*)()>(dlsym(RTLD_DEFAULT, "getPythonPBCRuntimeLibraryFunction"));
    using RuntimePath = std::string (*)();
    auto runtimePath = getter ? reinterpret_cast<RuntimePath>(getter()) : nullptr;
    if (!runtimePath) {
        return {};
    }
    try {
        std::string path = runtimePath();
        if (path.empty()) {
            return {};
        }
        // A sequence is loaded once per instruction. Keep only the last concrete
        // sequence per thread, bounded by maxGates, and key on every input bit.
        struct Cache {
            std::string path;
            uint64_t theta = 0;
            uint64_t epsilon = 0;
            std::shared_ptr<const Sequence> result;
        };
        static thread_local Cache cache;
        double angle = theta.getValueAsDouble(), error = epsilon.getValueAsDouble();
        uint64_t angleBits = std::bit_cast<uint64_t>(angle);
        uint64_t errorBits = std::bit_cast<uint64_t>(error);
        if (cache.result && cache.path == path && cache.theta == angleBits &&
            cache.epsilon == errorBits) {
            return cache.result;
        }
        auto library = llvm::sys::DynamicLibrary::getPermanentLibrary(path.c_str());
        if (!library.isValid()) {
            return {};
        }
        using GetSize = size_t (*)(double, double, bool);
        using GetGates = void (*)(size_t *, size_t *, size_t, size_t, size_t, double, double, bool);
        using GetPhase = double (*)(double, double, bool);
        auto size =
            reinterpret_cast<GetSize>(library.getAddressOfSymbol("rs_decomposition_get_size"));
        auto gates =
            reinterpret_cast<GetGates>(library.getAddressOfSymbol("rs_decomposition_get_gates"));
        auto phase =
            reinterpret_cast<GetPhase>(library.getAddressOfSymbol("rs_decomposition_get_phase"));
        if (!size || !gates || !phase) {
            return {};
        }
        size_t count = size(angle, error, true);
        if (count > maxGates) {
            return {};
        }
        Sequence result{std::vector<size_t>(count), phase(angle, error, true)};
        if (count) {
            gates(result.gates.data(), result.gates.data(), 0, count, 1, angle, error, true);
        }
        if (!std::isfinite(result.phase) ||
            llvm::any_of(result.gates, [](size_t gate) { return gate > 18; })) {
            return {};
        }
        cache = {std::move(path), angleBits, errorBits,
                 std::make_shared<const Sequence>(std::move(result))};
        return cache.result;
    } catch (...) {
        // Native synthesis can reject an unsupported precision or fail to find a
        // decomposition. Catch C++ exceptions here, before any language boundary.
        return {};
    }
}

} // namespace

bool isPBCSynthesisCall(Operation *op) {
    auto call = dyn_cast<func::CallOp>(op);
    if (!isRuntimeCall(call)) {
        return false;
    }
    return call.getCallee() != "rs_decomposition_get_gates" ||
           immutableFill(call.getOperand(0)) == call;
}

Attribute evaluatePBCSynthesisValue(Value value, llvm::function_ref<Attribute(Value)> resolve) {
    if (auto call = value.getDefiningOp<func::CallOp>(); isRuntimeCall(call)) {
        auto sequence = evaluateSequence(call.getOperands(), resolve);
        if (!sequence) {
            return {};
        }
        if (call.getCallee() == "rs_decomposition_get_size") {
            return IntegerAttr::get(value.getType(), sequence->gates.size());
        }
        return FloatAttr::get(value.getType(), sequence->phase);
    }
    auto load = value.getDefiningOp<memref::LoadOp>();
    if (!load || load.getIndices().size() != 1) {
        return {};
    }
    auto fill = immutableFill(load.getMemref());
    if (!fill) {
        return {};
    }
    auto index = dyn_cast_or_null<IntegerAttr>(resolve(load.getIndices().front()));
    auto alloc = load.getMemref().getDefiningOp<memref::AllocOp>();
    int64_t capacity = alloc.getType().getDimSize(0);
    if (ShapedType::isDynamic(capacity)) {
        auto size = dyn_cast_or_null<IntegerAttr>(resolve(alloc.getDynamicSizes().front()));
        if (!size) {
            return {};
        }
        capacity = size.getInt();
    }
    if (!index) {
        return {};
    }
    auto sequence = evaluateSequence(fill.getOperands().drop_front(), resolve);
    if (!sequence || capacity != static_cast<int64_t>(sequence->gates.size()) ||
        index.getInt() < 0 || index.getInt() >= capacity) {
        return {};
    }
    return IntegerAttr::get(value.getType(), sequence->gates[index.getInt()]);
}

} // namespace catalyst::pbc
