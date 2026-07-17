
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
#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h" // for gtest printing on failure
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
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
#include "mlir/Support/LLVM.h"

#include "Quantum/IR/QuantumDialect.h"
#include "Quantum/IR/QuantumInterfaces.h"
#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst::quantum;

/// The upstream MLIR Test dialect does not have a header we can include
/// We must declare the registration function, and link to the corresponding upstream target
/// in CMake.
namespace test {
void registerTestDialect(mlir::DialectRegistry &);
} // namespace test

TEST(DecomposableGateInterfaceTests, CustomOp)
{
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  %oq0, %oq1 = quantum.custom "RX"(%angle) %q0, %q1 : !quantum.bit, !quantum.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate customOp = *module->getOps<CustomOp>().begin();

    ASSERT_EQ(customOp.getOperatorName(), "RX");

    // This is needed to keep the backing array from being deleted
    llvm::SmallVector<mlir::Type, 1> backing({mlir::Float64Type::get(&context)});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(customOp.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(customOp.getWireLens(), std::vector<size_t>({2}));

    ASSERT_EQ(customOp.getStaticData().size(), 0);

    ASSERT_EQ(customOp.getGraphOpId(), "RX[f64][2]{}");
}

TEST(DecomposableGateInterfaceTests, MultiRZOp)
{
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  %q2 = quantum.alloc_qb : !quantum.bit
  %mrz:3 = quantum.multirz(%angle) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate multiRZ = *module->getOps<MultiRZOp>().begin();

    ASSERT_EQ(multiRZ.getOperatorName(), "MultiRZ");

    // This is needed to keep the backing array from being deleted
    llvm::SmallVector<mlir::Type, 1> backing({mlir::Float64Type::get(&context)});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(multiRZ.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(multiRZ.getWireLens(), std::vector<size_t>({3}));

    ASSERT_EQ(multiRZ.getStaticData().size(), 0);

    ASSERT_EQ(multiRZ.getGraphOpId(), "MultiRZ[f64][3]{}");
}

TEST(DecomposableGateInterfaceTests, PauliRotOp)
{
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  %q2 = quantum.alloc_qb : !quantum.bit
  %0:3 = quantum.paulirot ["X", "Y", "Z"] (%angle) %q0, %q1, %q2 : !quantum.bit, !quantum.bit, !quantum.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate paulirot = *module->getOps<PauliRotOp>().begin();

    ASSERT_EQ(paulirot.getOperatorName(), "PauliRot");

    // This is needed to keep the backing array from being deleted
    llvm::SmallVector<mlir::Type, 1> backing({mlir::Float64Type::get(&context)});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(paulirot.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(paulirot.getWireLens(), std::vector<size_t>({3}));

    mlir::NamedAttribute entry(mlir::StringAttr::get(&context, "pauli_word"),
                               mlir::StringAttr::get(&context, "XYZ"));
    mlir::DictionaryAttr expectedStaticData = mlir::DictionaryAttr::get(&context, {entry});
    ASSERT_EQ(paulirot.getStaticData(), expectedStaticData);

    ASSERT_EQ(paulirot.getGraphOpId(), "PauliRot[f64][3]{pauli_word:XYZ}");
}

TEST(DecomposableGateInterfaceTests, PCPhaseOP)
{
    std::string moduleStr = R"mlir(
module {
  %theta = arith.constant 3.7 : f64
  %dim = arith.constant 42.0 : f64
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  %q2 = quantum.alloc_qb : !quantum.bit
  %oq0, %oq1, %oq2 = quantum.pcphase(%theta, %dim) %q0, %q1 ctrls(%q2) : !quantum.bit, !quantum.bit ctrls !quantum.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate pcphase = *module->getOps<PCPhaseOp>().begin();

    ASSERT_EQ(pcphase.getOperatorName(), "PCPhase");

    // This is needed to keep the backing array from being deleted
    Type f64Type = mlir::Float64Type::get(&context);
    llvm::SmallVector<mlir::Type, 2> backing({f64Type, f64Type});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(pcphase.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    // Controls are not part of the gate wires considered by the decomp interface
    ASSERT_EQ(pcphase.getWireLens(), std::vector<size_t>({2}));

    ASSERT_EQ(pcphase.getStaticData().size(), 0);

    ASSERT_EQ(pcphase.getGraphOpId(), "PCPhase[f64,f64][2]{}");
}

TEST(DecomposableGateInterfaceTests, GlobalPhaseOp)
{
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  quantum.gphase(%angle)
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate gphase = *module->getOps<GlobalPhaseOp>().begin();

    ASSERT_EQ(gphase.getOperatorName(), "GlobalPhase");

    // This is needed to keep the backing array from being deleted
    llvm::SmallVector<mlir::Type, 1> backing({mlir::Float64Type::get(&context)});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(gphase.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(gphase.getWireLens(), std::vector<size_t>({0}));

    ASSERT_EQ(gphase.getStaticData().size(), 0);

    ASSERT_EQ(gphase.getGraphOpId(), "GlobalPhase[f64][0]{}");
}

TEST(DecomposableGateInterfaceTests, QubitUnitaryOp)
{
    std::string moduleStr = R"mlir(
module {
  %matrix = "test.op"() : () -> tensor<4x4xcomplex<f64>>
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  %q2 = quantum.alloc_qb : !quantum.bit
  %oq0, %oq1, %oq2 = quantum.unitary(%matrix : tensor<4x4xcomplex<f64>>) %q0, %q1 ctrls(%q2) : !quantum.bit, !quantum.bit ctrls !quantum.bit
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QuantumDialect>();
    test::registerTestDialect(registry);
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate unitary = *module->getOps<QubitUnitaryOp>().begin();

    ASSERT_EQ(unitary.getOperatorName(), "QubitUnitary");

    // This is needed to keep the backing array from being deleted
    Type f64type = mlir::Float64Type::get(&context);
    Type tensorType = mlir::RankedTensorType::get({4, 4}, mlir::ComplexType::get(f64type));
    llvm::SmallVector<mlir::Type, 1> backing({tensorType});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(unitary.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(unitary.getWireLens(), std::vector<size_t>({2}));

    ASSERT_EQ(unitary.getStaticData().size(), 0);

    ASSERT_EQ(unitary.getGraphOpId(), "QubitUnitary["
                                      "[[complex<f64>,complex<f64>,complex<f64>,complex<f64>],"
                                      "[complex<f64>,complex<f64>,complex<f64>,complex<f64>],"
                                      "[complex<f64>,complex<f64>,complex<f64>,complex<f64>],"
                                      "[complex<f64>,complex<f64>,complex<f64>,complex<f64>]]"
                                      "][2]{}");
}

TEST(DecomposableGateInterfaceTests, OperatorOpQubits)
{
    std::string moduleStr = R"mlir(
module {
  %angle = arith.constant 3.1 : f64
  %flag = arith.constant 0 : i1
  %index = arith.constant 5 : i64
  %q0 = quantum.alloc_qb : !quantum.bit
  %q1 = quantum.alloc_qb : !quantum.bit
  %0:2 = quantum.operator "testInterfaceOp"(%flag: i1, %angle: f64, %index: i64) qubits(%q0, %q1) static_data = {"myStaticArray"=[1,2,3], "myStaticString"="Test", "myStaticInt"=4}
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    auto operators = module->getOps<OperatorOp>();
    DecomposableGate op = *operators.begin();

    ASSERT_EQ(op.getOperatorName(), "testInterfaceOp");

    // This is needed to keep the backing array from being deleted
    llvm::SmallVector<mlir::Type, 1> backing({mlir::IntegerType::get(&context, 1),
                                              mlir::Float64Type::get(&context),
                                              mlir::IntegerType::get(&context, 64)});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(op.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(op.getWireLens(), std::vector<size_t>({2}));

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

    ASSERT_EQ(
        op.getGraphOpId(),
        "testInterfaceOp[i1,f64,i64][2]{myStaticArray:[1,2,3],myStaticInt:4,myStaticString:Test}");
}

TEST(DecomposableGateInterfaceTests, OperatorOpQureg)
{
    std::string moduleStr = R"mlir(
func.func @testfunc(%first : tensor<1xi64>, %secondthird : tensor<2xi64>) {
  %angle = arith.constant 3.1 : f64
  %flag = arith.constant 0 : i1
  %index = arith.constant 5 : i64

  %reg = quantum.alloc(4) : !quantum.reg

  %0 = quantum.operator "testOperatorQreg"(%flag: i1, %angle: f64, %index: i64) quregs(%reg) indices(%first: tensor<1xi64>, %secondthird: tensor<2xi64>) static_data={"myStaticArray"=[4,2.4,4], "myStaticString"="string", "myStaticInt"=8}
  return
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate op;
    module->walk([&](OperatorOp walkOp) { op = walkOp; });

    ASSERT_EQ(op.getOperatorName(), "testOperatorQreg");

    // This is needed to keep the backing array from being deleted
    llvm::SmallVector<mlir::Type, 1> backing({mlir::IntegerType::get(&context, 1),
                                              mlir::Float64Type::get(&context),
                                              mlir::IntegerType::get(&context, 64)});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(op.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(op.getWireLens(), std::vector<size_t>({1, 2}));

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
              "testOperatorQreg[i1,f64,i64][1,2]{myStaticArray:[4,2.400000e+00,4],"
              "myStaticInt:8,myStaticString:string}");
}

TEST(DecomposableGateInterfaceTests, OperatorOpUID)
{
    std::string moduleStr = R"mlir(
func.func @testfunc(%first : tensor<1xi64>, %secondthird : tensor<2xi64>) {
  %angle = arith.constant 3.1 : f64
  %flag = arith.constant 0 : i1
  %index = arith.constant 5 : i64

  %reg = quantum.alloc(4) : !quantum.reg
  %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit

  %0 = quantum.operator "testOperatorUID"(%flag: i1, %angle: f64, %index: i64)
    UID(248) quregs(%reg) indices(%first: tensor<1xi64>, %secondthird: tensor<2xi64>)
  return
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate op;
    module->walk([&](OperatorOp walkOp) { op = walkOp; });

    ASSERT_EQ(op.getOperatorName(), "testOperatorUID");

    // This is needed to keep the backing array from being deleted
    llvm::SmallVector<mlir::Type, 1> backing({mlir::IntegerType::get(&context, 1),
                                              mlir::Float64Type::get(&context),
                                              mlir::IntegerType::get(&context, 64)});
    mlir::TypeRange expectedDynamicShape(backing);
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(op.getDynamicShape()),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(op.getWireLens(), std::vector<size_t>({1, 2}));

    ASSERT_EQ(op.getStaticData(), mlir::DictionaryAttr::get(&context, {}));

    ASSERT_EQ(op.getGraphOpId(), "testOperatorUID[i1,f64,i64][1,2]{}[248]");
}

TEST(DecomposableGateInterfaceTests, OperatorOpNestedArgs)
{
    std::string moduleStr = R"mlir(
func.func @testfunc(%arg: tensor<3xi64>, %arg2: tensor<2xf64>) {

  %reg = quantum.alloc(4) : !quantum.reg
  %q0 = quantum.extract %reg[0] : !quantum.reg -> !quantum.bit

  %0 = quantum.operator "testNestedArgs"(%arg : tensor<3xi64>, %arg2: tensor<2xf64>)
    qubits(%q0)
  return
}
    )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, QuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);

    DecomposableGate op;
    module->walk([&](OperatorOp walkOp) { op = walkOp; });

    ASSERT_EQ(op.getOperatorName(), "testNestedArgs");

    // This is needed to keep the backing array from being deleted
    mlir::Type int64 = mlir::IntegerType::get(&context, 64);
    llvm::SmallVector<mlir::Type, 1> backing(
        {mlir::RankedTensorType::get(ArrayRef<int64_t>({3}), int64),
         mlir::RankedTensorType::get(ArrayRef<int64_t>({2}), mlir::Float64Type::get(&context))});
    mlir::TypeRange expectedDynamicShape(backing);
    mlir::TypeRange actual(op.getDynamicShape());
    ASSERT_EQ(llvm::SmallVector<mlir::Type>(actual),
              llvm::SmallVector<mlir::Type>(expectedDynamicShape));

    ASSERT_EQ(op.getWireLens(), std::vector<size_t>({1}));

    ASSERT_EQ(op.getStaticData(), mlir::DictionaryAttr::get(&context, {}));

    ASSERT_EQ(op.getGraphOpId(), "testNestedArgs[[i64,i64,i64],[f64,f64]][1]{}");
}
