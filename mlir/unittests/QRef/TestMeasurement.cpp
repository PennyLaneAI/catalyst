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

#include "QRef/IR/QRefOps.h"

using namespace mlir;
using namespace catalyst;

TEST(MeasurementTests, BuildersAndInterfaces) {
    DialectRegistry registry;
    registry.insert<func::FuncDialect, qref::QRefDialect>();
    MLIRContext context(registry);
    auto module = parseSourceString<ModuleOp>(R"mlir(
        func.func @build(%angle: f64, %indices: tensor<?xi64>,
                         %q: !qref.bit, %reg: !qref.reg<?>) {
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

    auto rQubit = qref::MeasurementOp::create(builder, loc, mres, "Basis", ValueRange{args[0]},
                                              ValueRange{args[2]}, data);
    auto rReg = qref::MeasurementOp::create(builder, loc, TypeRange{}, "Register", ValueRange{},
                                            args[3], ValueRange{args[1]});

    ASSERT_TRUE(succeeded(verify(*module)));
    EXPECT_EQ(rQubit.getNumResults(), 2u);
    EXPECT_EQ(rQubit.getStaticData(), data);
    EXPECT_EQ(rReg.getNumResults(), 0u);
    EXPECT_TRUE(rReg.getStaticData().empty());

    auto resource = cast<ResourceQuantumOpInterface>(rQubit.getOperation());
    EXPECT_EQ(resource.getResourceName(), "Basis");
    EXPECT_EQ(resource.getResourceNumQubits(), 1u);
    EXPECT_EQ(resource.getResourceNumCtrlQubits(), 0u);
    EXPECT_EQ(resource.getResourceNumParams(), 1u);
    EXPECT_FALSE(resource.getResourceAdjointFlag());

    for (Operation *op : {rQubit.getOperation(), rReg.getOperation()}) {
        EXPECT_TRUE(isa<quantum::MeasurementProcess>(op));
        EXPECT_FALSE(isa<quantum::TerminalMeasurement>(op));
        EXPECT_FALSE((isa<quantum::QuantumOperation, qref::QuantumOperation>(op)));
        EXPECT_FALSE(isMemoryEffectFree(op));
    }
}
