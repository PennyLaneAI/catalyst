// Copyright 2025 Xanadu Quantum Technologies Inc.
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

#pragma once

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

#include "RTIO/IR/RTIOOps.h"

namespace catalyst {
namespace rtio {

/// Helpers for reading nested keys from module rtio.config (device_db JSON).
namespace device_db_detail {

/// Descend one level in nested DictionaryAttr / ConfigAttr.
inline mlir::Attribute descendByKey(mlir::Attribute parent, llvm::StringRef key)
{
    if (!parent) {
        return {};
    }
    if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(parent)) {
        return dict.get(key);
    }
    if (auto cfg = mlir::dyn_cast<ConfigAttr>(parent)) {
        return cfg.get(key);
    }
    return {};
}

/// Follow a chain of keys from root. Returns null if any step is missing or not a container.
inline mlir::Attribute walkAttrPath(mlir::Attribute root, llvm::ArrayRef<llvm::StringRef> path)
{
    mlir::Attribute current = root;
    for (llvm::StringRef key : path) {
        current = descendByKey(current, key);
        if (!current) {
            return {};
        }
    }
    return current;
}

/// Integer at path
inline int64_t intAtPath(mlir::Attribute root, llvm::ArrayRef<llvm::StringRef> path)
{
    mlir::Attribute leaf = walkAttrPath(root, path);
    if (auto intAttr = mlir::dyn_cast_or_null<mlir::IntegerAttr>(leaf)) {
        return intAttr.getInt();
    }
    return 0;
}

} // namespace device_db_detail

/// Extract the static channel ID from an RTIO channel type.
inline int32_t extractChannelId(mlir::Value channelValue)
{
    auto type = mlir::cast<ChannelType>(channelValue.getType());

    assert(type.isStatic() && "Only static channel IDs are supported");
    return type.getChannelId().getInt();
}

/// Compute the device address for a given channel value.
inline mlir::Value computeChannelDeviceAddr(mlir::OpBuilder &builder, mlir::Operation *op,
                                            mlir::Value channelValue)
{
    mlir::Location loc = op->getLoc();
    mlir::ModuleOp mod = op->getParentOfType<mlir::ModuleOp>();
    auto configAttr = mod->getAttrOfType<ConfigAttr>(ConfigAttr::getModuleAttrName());
    assert(configAttr && "configAttr not found");

    mlir::Attribute leaf = device_db_detail::walkAttrPath(
        configAttr, {"device_db", "ttl_urukul0_sw0", "arguments", "channel"});
    assert(leaf && "device_db.ttl_urukul0_sw0.arguments.channel missing");
    int64_t channelBase = mlir::cast<mlir::IntegerAttr>(leaf).getInt();

    llvm::APInt channelIdAPInt;
    assert(mlir::matchPattern(channelValue, mlir::m_ConstantInt(&channelIdAPInt)) &&
           "only static channels are supported");
    int64_t channelId = channelIdAPInt.getSExtValue();
    int32_t addr = static_cast<int32_t>((channelId + channelBase) << 8);
    return mlir::arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(addr));
}

} // namespace rtio
} // namespace catalyst
