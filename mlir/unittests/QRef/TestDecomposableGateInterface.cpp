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

#include <array>
#include <cstddef>
#include <string>

#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Format.h" // for gtest printing on failure
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"

#include "QRef/IR/QRefDialect.h"
#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"

using namespace mlir;
using namespace catalyst::qref;

/// The upstream MLIR Test dialect does not have a header we can include
/// We must declare the registration function, and link to the corresponding upstream target
/// in CMake.
namespace test {
void registerTestDialect(mlir::DialectRegistry &);
} // namespace test

TEST(DecomposableGateInterfaceTests, CustomOp) {
    std::string moduleStr = R"mlir(
module {
    %angle = arith.constant 3.1 : f64
    %q0 = qref.alloc_qb : !qref.bit
    %q1 = qref.alloc_qb : !qref.bit
    qref.custom "RX"(%angle) %q0, %q1 : !qref.bit, !qref.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate customOp = *module->getOps<CustomOp>().begin();

    ASSERT_EQ(customOp.getOperatorName(), "RX");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"0", {Float64Type::get(&context)}}};
    ASSERT_EQ(customOp.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"wires", 2}};
    ASSERT_EQ(customOp.getWireLens(), expectedWires);

    ASSERT_EQ(customOp.getStaticData().size(), 0);

    ASSERT_EQ(customOp.getGraphOpId(), "RX{0:[f64]}{wires:2}{}");
}

TEST(DecomposableGateInterfaceTests, MultiRZOp) {
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %q0 = qref.alloc_qb : !qref.bit
  %q1 = qref.alloc_qb : !qref.bit
  %q2 = qref.alloc_qb : !qref.bit
  qref.multirz(%angle) %q0, %q1, %q2 : !qref.bit, !qref.bit, !qref.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate multiRZ = *module->getOps<MultiRZOp>().begin();

    ASSERT_EQ(multiRZ.getOperatorName(), "MultiRZ");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"theta", {Float64Type::get(&context)}}};
    ASSERT_EQ(multiRZ.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"wires", 3}};
    ASSERT_EQ(multiRZ.getWireLens(), expectedWires);

    ASSERT_EQ(multiRZ.getStaticData().size(), 0);

    ASSERT_EQ(multiRZ.getGraphOpId(), "MultiRZ{theta:[f64]}{wires:3}{}");
}

TEST(DecomposableGateInterfaceTests, PauliRotOp) {
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %q0 = qref.alloc_qb : !qref.bit
  %q1 = qref.alloc_qb : !qref.bit
  %q2 = qref.alloc_qb : !qref.bit
  qref.paulirot ["X", "Y", "Z"] (%angle) %q0, %q1, %q2 : !qref.bit, !qref.bit, !qref.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate paulirot = *module->getOps<PauliRotOp>().begin();

    ASSERT_EQ(paulirot.getOperatorName(), "PauliRot");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"theta", {Float64Type::get(&context)}}};
    ASSERT_EQ(paulirot.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"wires", 3}};
    ASSERT_EQ(paulirot.getWireLens(), expectedWires);

    mlir::NamedAttribute entry(mlir::StringAttr::get(&context, "pauli_word"),
                               mlir::StringAttr::get(&context, "XYZ"));
    mlir::DictionaryAttr expectedStaticData = mlir::DictionaryAttr::get(&context, {entry});
    ASSERT_EQ(paulirot.getStaticData(), expectedStaticData);

    ASSERT_EQ(paulirot.getGraphOpId(), "PauliRot{theta:[f64]}{wires:3}{pauli_word:XYZ}");
}

TEST(DecomposableGateInterfaceTests, PCPhaseOP) {
    std::string moduleStr = R"mlir(
module {
  %theta = arith.constant 3.7 : f64
  %q0 = qref.alloc_qb : !qref.bit
  %q1 = qref.alloc_qb : !qref.bit
  %q2 = qref.alloc_qb : !qref.bit
  qref.pcphase(%theta, dim : 0) %q0, %q1 ctrls(%q2) : !qref.bit, !qref.bit ctrls !qref.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate pcphase = *module->getOps<PCPhaseOp>().begin();

    ASSERT_EQ(pcphase.getOperatorName(), "PCPhase");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"phi", {Float64Type::get(&context)}}};
    ASSERT_EQ(pcphase.getDynamicShape(), expectedDynamicShape);

    // Controls are not part of the gate wires considered by the decomp interface
    llvm::StringMap<size_t> expectedWires = {{"wires", 2}};
    ASSERT_EQ(pcphase.getWireLens(), expectedWires);

    mlir::NamedAttribute entry(mlir::StringAttr::get(&context, "dim"),
                               mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 0));
    mlir::DictionaryAttr expectedStaticData = mlir::DictionaryAttr::get(&context, {entry});
    ASSERT_EQ(pcphase.getStaticData(), expectedStaticData);

    ASSERT_EQ(pcphase.getGraphOpId(), "PCPhase{phi:[f64]}{wires:2}{dim:0}");
}

TEST(DecomposableGateInterfaceTests, GlobalPhaseOp) {
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  qref.gphase(%angle)
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate gphase = *module->getOps<GlobalPhaseOp>().begin();

    ASSERT_EQ(gphase.getOperatorName(), "GlobalPhase");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"phi", {mlir::Float64Type::get(&context)}}};
    ASSERT_EQ(gphase.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {};
    ASSERT_EQ(gphase.getWireLens(), expectedWires);

    ASSERT_EQ(gphase.getStaticData().size(), 0);

    ASSERT_EQ(gphase.getGraphOpId(), "GlobalPhase{phi:[f64]}{}{}");
}

TEST(DecomposableGateInterfaceTests, ControlledGlobalPhaseOp) {
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %true = arith.constant true
  %q0 = qref.alloc_qb : !qref.bit
  qref.gphase(%angle) ctrls (%q0) ctrlvals (%true) : ctrls !qref.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate gphase = *module->getOps<GlobalPhaseOp>().begin();

    ASSERT_EQ(gphase.getOperatorName(), "GlobalPhase");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"phi", {mlir::Float64Type::get(&context)}}};
    ASSERT_EQ(gphase.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {};
    ASSERT_EQ(gphase.getWireLens(), expectedWires);

    ASSERT_EQ(gphase.getStaticData().size(), 0);

    ASSERT_EQ(gphase.getGraphOpId(), "GlobalPhase{phi:[f64]}{}{}");
}

TEST(DecomposableGateInterfaceTests, QubitUnitaryOp) {
    std::string moduleStr = R"mlir(
module {
  %matrix = "test.op"() : () -> tensor<4x4xcomplex<f64>>
  %q0 = qref.alloc_qb : !qref.bit
  %q1 = qref.alloc_qb : !qref.bit
  %q2 = qref.alloc_qb : !qref.bit
  qref.unitary(%matrix : tensor<4x4xcomplex<f64>>) %q0, %q1 ctrls(%q2) : !qref.bit, !qref.bit ctrls !qref.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    test::registerTestDialect(registry);
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate unitary = *module->getOps<QubitUnitaryOp>().begin();

    ASSERT_EQ(unitary.getOperatorName(), "QubitUnitary");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"U",
         {mlir::RankedTensorType::get({4, 4},
                                      mlir::ComplexType::get(mlir::Float64Type::get(&context)))}}};
    ASSERT_EQ(unitary.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"wires", 2}};
    ASSERT_EQ(unitary.getWireLens(), expectedWires);

    ASSERT_EQ(unitary.getStaticData().size(), 0);

    ASSERT_EQ(unitary.getGraphOpId(), "QubitUnitary{U:[tensor<4x4xcomplex<f64>>]}{wires:2}{}");
}

TEST(DecomposableGateInterfaceTests, OperatorOpQubits) {
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %flag = arith.constant 0 : i1
  %index = arith.constant 5 : i64
  %q0 = qref.alloc_qb : !qref.bit
  %q1 = qref.alloc_qb : !qref.bit
  qref.operator "testInterfaceOp"(%flag: i1, %angle: f64, %index: i64) qubits(%q0, %q1) static_data = {"myStaticArray"=[1,2,3], "myStaticString"="Test", "myStaticInt"=4} param_map = {flag = [0], angle = [1], index = [2]} qubit_map = {wire1 = [0], wire2 = [1]}
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    auto operators = module->getOps<OperatorOp>();
    DecomposableGate op = *operators.begin();

    ASSERT_EQ(op.getOperatorName(), "testInterfaceOp");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"flag", {mlir::IntegerType::get(&context, 1)}},
        {"angle", {mlir::Float64Type::get(&context)}},
        {"index", {mlir::IntegerType::get(&context, 64)}}};
    ASSERT_EQ(op.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"wire1", 1}, {"wire2", 1}};
    ASSERT_EQ(op.getWireLens(), expectedWires);

    IntegerType i64 = IntegerType::get(&context, 64);
    llvm::SmallVector<mlir::Attribute> arr({
        mlir::IntegerAttr::get(i64, 1),
        mlir::IntegerAttr::get(i64, 2),
        mlir::IntegerAttr::get(i64, 3),
    });
    mlir::NamedAttribute arrAttr(mlir::StringAttr::get(&context, "myStaticArray"),
                                 mlir::ArrayAttr::get(&context, arr));

    mlir::NamedAttribute stringAttr(mlir::StringAttr::get(&context, "myStaticString"),
                                    mlir::StringAttr::get(&context, "Test"));

    mlir::NamedAttribute intAttr(mlir::StringAttr::get(&context, "myStaticInt"),
                                 mlir::IntegerAttr::get(i64, 4));
    mlir::DictionaryAttr expectedStaticData =
        mlir::DictionaryAttr::get(&context, {arrAttr, stringAttr, intAttr});
    ASSERT_EQ(op.getStaticData(), expectedStaticData);

    ASSERT_EQ(op.getGraphOpId(),
              "testInterfaceOp{angle:[f64],flag:[i1],index:[i64]}{wire1:1,wire2:1}{"
              "myStaticArray:[1,2,3],myStaticInt:4,myStaticString:Test}");
}

TEST(DecomposableGateInterfaceTests, OperatorOpGOIDTypeConflict) {
    std::string moduleStr = R"mlir(
module {
  %op0 = "test.op0"() : () -> tensor<f64>
  %op1 = "test.op1"() : () -> f64
  %q0 = qref.alloc_qb : !qref.bit
  %q1 = qref.alloc_qb : !qref.bit
  qref.operator "testOperator"(%op0: tensor<f64>) qubits(%q0, %q1) param_map = {op=[0]} qubit_map = {wire1=[0], wire2=[1]}
  qref.operator "testOperator"(%op1: f64) qubits(%q0, %q1) param_map = {op=[0]} qubit_map = {wire1=[0], wire2=[1]}
}
    )mlir";
    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QRefDialect>();
    test::registerTestDialect(registry);
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    // Obtain DecomposableGate from the two OperatorOps being compared
    std::array<DecomposableGate, 2> ops;
    int i = 0;
    for (auto op : module->getOps<OperatorOp>()) {
        ops[i] = op;
        i++;
    }

    // Ensure the two ops do not have the same GOID
    ASSERT_NE(ops[0].getGraphOpId(), ops[1].getGraphOpId());
}

TEST(DecomposableGateInterfaceTests, OperatorOpQureg) {
    std::string moduleStr = R"mlir(
func.func @testfunc(%first : tensor<1xi64>, %secondthird : tensor<2xi64>) {
  %angle = arith.constant 3.1 : f64
  %flag = arith.constant 0 : i1
  %index = arith.constant 5 : i64

  %reg = qref.alloc(4) : !qref.reg<4>

  qref.operator "testOperatorQureg"(%flag: i1, %angle: f64, %index: i64) quregs(%reg : !qref.reg<4>) indices(%first: tensor<1xi64>, %secondthird: tensor<2xi64>) static_data={"myStaticArray"=[4,2.4,4], "myStaticString"="string", "myStaticInt"=8} param_map = {angle=[1], index=[2], flag=[0]} qubit_map = {reg=[0, 1]} 
  return
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate op;
    module->walk([&](OperatorOp walkOp) { op = walkOp; });

    ASSERT_EQ(op.getOperatorName(), "testOperatorQureg");

    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"flag", {mlir::IntegerType::get(&context, 1)}},
        {"angle", {mlir::Float64Type::get(&context)}},
        {"index", {mlir::IntegerType::get(&context, 64)}}};
    ASSERT_EQ(op.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"reg", 3}};
    ASSERT_EQ(op.getWireLens(), expectedWires);

    IntegerType i64 = IntegerType::get(&context, 64);
    Float64Type f64 = mlir::Float64Type::get(&context);
    llvm::SmallVector<mlir::Attribute> arr({
        mlir::IntegerAttr::get(i64, 4),
        mlir::FloatAttr::get(f64, 2.4),
        mlir::IntegerAttr::get(i64, 4),
    });
    mlir::NamedAttribute arrAttr(mlir::StringAttr::get(&context, "myStaticArray"),
                                 mlir::ArrayAttr::get(&context, arr));

    mlir::NamedAttribute stringAttr(mlir::StringAttr::get(&context, "myStaticString"),
                                    mlir::StringAttr::get(&context, "string"));

    mlir::NamedAttribute intAttr(mlir::StringAttr::get(&context, "myStaticInt"),
                                 mlir::IntegerAttr::get(i64, 8));
    mlir::DictionaryAttr expectedStaticData =
        mlir::DictionaryAttr::get(&context, {arrAttr, stringAttr, intAttr});
    ASSERT_EQ(op.getStaticData(), expectedStaticData);

    ASSERT_EQ(op.getGraphOpId(),
              "testOperatorQureg{angle:[f64],flag:[i1],index:[i64]}{reg:3}{"
              "myStaticArray:[4,2.400000e+00,4],myStaticInt:8,myStaticString:string}");
}

TEST(DecomposableGateInterfaceTests, OperatorOpUID) {
    std::string moduleStr = R"mlir(
func.func @testfunc(%first : tensor<1xi64>, %secondthird : tensor<2xi64>) {
  %angle = arith.constant 3.1 : f64
  %flag = arith.constant 0 : i1
  %index = arith.constant 5 : i64

  %reg = qref.alloc(4) : !qref.reg<4>

  qref.operator "testOperatorUID"(%flag: i1, %angle: f64, %index: i64)
    UID(248) quregs(%reg : !qref.reg<4>) indices(%first: tensor<1xi64>, %secondthird: tensor<2xi64>) param_map = {flag=[0], angle=[1], index=[2]} qubit_map = {reg=[0, 1]}
  return
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate op;
    module->walk([&](OperatorOp walkOp) { op = walkOp; });

    ASSERT_EQ(op.getOperatorName(), "testOperatorUID");

    // This is needed to keep the backing array from being deleted
    // llvm::SmallVector<llvm::SmallVector<mlir::Type>, 1> backing(
    //     {mlir::IntegerType::get(&context, 1), mlir::Float64Type::get(&context),
    //      mlir::IntegerType::get(&context, 64)});
    llvm::StringMap<llvm::SmallVector<mlir::Type>> expectedDynamicShape = {
        {"flag", {mlir::IntegerType::get(&context, 1)}},
        {"angle", {mlir::Float64Type::get(&context)}},
        {"index", {mlir::IntegerType::get(&context, 64)}}};
    ASSERT_EQ(op.getDynamicShape(), expectedDynamicShape);

    llvm::StringMap<size_t> expectedWires = {{"reg", 3}};
    ASSERT_EQ(op.getWireLens(), expectedWires);

    ASSERT_EQ(op.getStaticData(), mlir::DictionaryAttr::get(&context, {}));

    ASSERT_EQ(op.getGraphOpId(),
              "testOperatorUID{angle:[f64],flag:[i1],index:[i64]}{reg:3}{}[248]");
}
