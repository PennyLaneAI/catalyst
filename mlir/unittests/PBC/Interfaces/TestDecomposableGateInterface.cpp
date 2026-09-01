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

#include <cstddef>
#include <string>

#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Format.h" // for gtest printing on failure
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"

#include "PBC/IR/PBCDialect.h"
#include "PBC/IR/PBCOpInterfaces.h"
#include "PBC/IR/PBCOps.h"
#include "Quantum/IR/QuantumDialect.h"

using namespace mlir;
using namespace catalyst::pbc;

TEST(DecomposableGateInterfaceTests, PPRotationOp) {
    std::string moduleStr = R"mlir(
module {
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  %q2 = quantum.alloc_qb : !quantum.bit
  %0:3 = pbc.ppr ["X", "Y", "Z"](4) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
}
    )mlir";

    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, catalyst::quantum::QuantumDialect, PBCDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate ppr = *module->getOps<PPRotationOp>().begin();

    ASSERT_EQ(ppr.getOperatorName(), "PauliRot");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"theta", {Float64Type::get(&context)}}};
    ASSERT_EQ(ppr.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"wires", 3}};
    ASSERT_EQ(ppr.getWireLens(), expectedWires);

    mlir::NamedAttribute entry(mlir::StringAttr::get(&context, "pauli_word"),
                               mlir::StringAttr::get(&context, "XYZ"));
    mlir::DictionaryAttr expectedStaticData = mlir::DictionaryAttr::get(&context, {entry});
    ASSERT_EQ(ppr.getStaticData(), expectedStaticData);

    ASSERT_EQ(ppr.getGraphOpId(), "PauliRot{theta:[f64]}{wires:3}{pauli_word:XYZ}");
}

TEST(DecomposableGateInterfaceTests, PPRotationArbitraryOp) {
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  %q2 = quantum.alloc_qb : !quantum.bit
  %0:3 = pbc.ppr.arbitrary ["X", "Y", "Z"](%angle) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
}
    )mlir";

    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, catalyst::quantum::QuantumDialect, PBCDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate ppr = *module->getOps<PPRotationArbitraryOp>().begin();

    ASSERT_EQ(ppr.getOperatorName(), "PauliRot");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"theta", {Float64Type::get(&context)}}};
    ASSERT_EQ(ppr.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"wires", 3}};
    ASSERT_EQ(ppr.getWireLens(), expectedWires);

    mlir::NamedAttribute entry(mlir::StringAttr::get(&context, "pauli_word"),
                               mlir::StringAttr::get(&context, "XYZ"));
    mlir::DictionaryAttr expectedStaticData = mlir::DictionaryAttr::get(&context, {entry});
    ASSERT_EQ(ppr.getStaticData(), expectedStaticData);

    ASSERT_EQ(ppr.getGraphOpId(), "PauliRot{theta:[f64]}{wires:3}{pauli_word:XYZ}");
}
