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

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"

#include "Quantum/IR/QuantumOps.h"

using namespace mlir;
using namespace catalyst;

TEST(MeasurementTests, BuildersAndInterfaces) {
    DialectRegistry registry;
    registry.insert<func::FuncDialect, quantum::QuantumDialect>();
    MLIRContext context(registry);
    auto module = parseSourceString<ModuleOp>(R"mlir(
        func.func @build(%angle: f64, %indices: tensor<?xi64>,
                         %q0: !quantum.bit, %q1: !quantum.bit, %reg: !quantum.reg) {
            return
        }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    auto func = *module->getOps<func::FuncOp>().begin();
    auto args = func.getArguments();
    OpBuilder builder = OpBuilder::atBlockBegin(&func.getBody().front());
    auto loc = builder.getUnknownLoc();
    SmallVector<Type> mres{builder.getI1Type(), builder.getF64Type()};
    auto data =
        builder.getDictionaryAttr({builder.getNamedAttr("plane", builder.getStringAttr("XY"))});

    auto qubits = quantum::MeasurementOp::create(builder, loc, mres, "Basis", ValueRange{args[0]},
                                                 ValueRange{args[2], args[3]}, data);
    auto reg = quantum::MeasurementOp::create(builder, loc, mres, "Register", ValueRange{}, args[4],
                                              ValueRange{args[1]});
    auto discardQubits = quantum::MeasurementOp::create(builder, loc, TypeRange{}, "Discard",
                                                        ValueRange{}, qubits.getOutQubits());
    auto discardReg = quantum::MeasurementOp::create(
        builder, loc, TypeRange{}, "Discard", ValueRange{}, reg.getOutQreg(), ValueRange{args[1]});

    ASSERT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(qubits.getNumResults(), 4u);
    ASSERT_EQ(qubits.getMres().size(), 2u);
    EXPECT_EQ(qubits.getMres()[0], qubits.getResult(0));
    EXPECT_EQ(qubits.getMres()[1], qubits.getResult(1));
    EXPECT_EQ(qubits.getMres()[0].getType(), mres[0]);
    EXPECT_EQ(qubits.getMres()[1].getType(), mres[1]);
    ASSERT_EQ(qubits.getOutQubits().size(), 2u);
    EXPECT_EQ(qubits.getOutQubits()[0], qubits.getResult(2));
    EXPECT_EQ(qubits.getOutQubits()[1], qubits.getResult(3));
    EXPECT_EQ(qubits.getOutQubits()[0].getType(), args[2].getType());
    EXPECT_EQ(qubits.getOutQubits()[1].getType(), args[3].getType());
    EXPECT_FALSE(qubits.getOutQreg());
    EXPECT_EQ(qubits.getStaticData(), data);

    EXPECT_EQ(reg.getNumResults(), 3u);
    ASSERT_EQ(reg.getMres().size(), 2u);
    EXPECT_EQ(reg.getMres()[0], reg.getResult(0));
    EXPECT_EQ(reg.getMres()[1], reg.getResult(1));
    EXPECT_EQ(reg.getMres()[0].getType(), mres[0]);
    EXPECT_EQ(reg.getMres()[1].getType(), mres[1]);
    EXPECT_TRUE(reg.getOutQubits().empty());
    EXPECT_EQ(reg.getOutQreg(), reg.getResult(2));
    EXPECT_EQ(reg.getOutQreg().getType(), args[4].getType());
    EXPECT_TRUE(reg.getStaticData().empty());

    EXPECT_EQ(discardQubits.getNumResults(), 2u);
    EXPECT_TRUE(discardQubits.getMres().empty());
    EXPECT_EQ(discardQubits.getOutQubits().size(), 2u);
    EXPECT_FALSE(discardQubits.getOutQreg());
    EXPECT_TRUE(discardQubits.getStaticData().empty());
    EXPECT_EQ(discardReg.getNumResults(), 1u);
    EXPECT_TRUE(discardReg.getMres().empty());
    EXPECT_TRUE(discardReg.getOutQubits().empty());
    EXPECT_EQ(discardReg.getOutQreg(), discardReg.getResult(0));
    EXPECT_TRUE(discardReg.getStaticData().empty());

    auto resource = cast<ResourceQuantumOpInterface>(qubits.getOperation());
    EXPECT_EQ(resource.getResourceName(), "Basis");
    EXPECT_EQ(resource.getResourceNumQubits(), 2u);
    EXPECT_EQ(resource.getResourceNumCtrlQubits(), 0u);
    EXPECT_EQ(resource.getResourceNumParams(), 1u);
    EXPECT_FALSE(resource.getResourceAdjointFlag());

    for (Operation *op : {qubits.getOperation(), reg.getOperation(), discardQubits.getOperation(),
                          discardReg.getOperation()}) {
        EXPECT_TRUE(isa<quantum::MeasurementProcess>(op));
        EXPECT_FALSE(isa<quantum::TerminalMeasurement>(op));
        EXPECT_FALSE(isa<quantum::QuantumOperation>(op));
        EXPECT_FALSE(isMemoryEffectFree(op));
    }
}

TEST(MeasurementTests, TerminalMeasurementInheritance) {
    DialectRegistry registry;
    registry.insert<func::FuncDialect, quantum::QuantumDialect>();
    MLIRContext context(registry);
    auto module = parseSourceString<ModuleOp>(R"mlir(
        func.func @terminals(%q: !quantum.bit) attributes {quantum.node} {
            %obs = quantum.compbasis qubits %q : !quantum.obs
            %samples = quantum.sample %obs : tensor<4x1xf64>
            %eigvals, %counts = quantum.counts %obs : tensor<2xf64>, tensor<2xi64>
            %expval = quantum.expval %obs : f64
            %variance = quantum.var %obs : f64
            %probs = quantum.probs %obs : tensor<2xf64>
            %state = quantum.state %obs : tensor<2xcomplex<f64>>
            return
        }
    )mlir",
                                              &context);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));

    Value observable;
    module->walk([&](quantum::ComputationalBasisOp op) { observable = op.getObs(); });
    ASSERT_TRUE(observable);

    unsigned terminalCount = 0;
    module->walk([&](quantum::TerminalMeasurement op) {
        EXPECT_TRUE(isa<quantum::MeasurementProcess>(op.getOperation()));
        quantum::MeasurementProcess process = op;
        EXPECT_EQ(process.getOperation(), op.getOperation());
        EXPECT_EQ(op.getObs(), observable);
        ++terminalCount;
    });
    EXPECT_EQ(terminalCount, 6u);

    unsigned measurementCount = 0;
    module->walk([&](quantum::MeasurementProcess) { ++measurementCount; });
    EXPECT_EQ(measurementCount, terminalCount);
}
