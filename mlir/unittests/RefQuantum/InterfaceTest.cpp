// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>

#include "gtest/gtest.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/SourceMgr.h"

#include "RefQuantum/IR/RefQuantumInterfaces.h"
#include "RefQuantum/IR/RefQuantumOps.h"

using namespace mlir;

namespace {

TEST(InterfaceTests, Getters)
{
    std::string moduleStr = R"mlir(
func.func @f(%w0: i64, %w1: i64, %param: f64, %bool: i1) {
    ref_quantum.custom "Rot"(%param, %param) %w0 adj ctrls (%w1) ctrlvals (%bool) : i64 ctrls i64
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::ref_quantum::RefQuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::ref_quantum::CustomOp customOp = *f.getOps<catalyst::ref_quantum::CustomOp>().begin();

    Block &bb = f.getCallableRegion()->front();
    auto args = bb.getArguments();

    // Run checks
    std::vector<Value> wireOperands = customOp.getWireOperands();
    ASSERT_TRUE(wireOperands.size() == 2 && wireOperands[0] == args[0] &&
                wireOperands[1] == args[1]);

    ValueRange nonCtrlWireOperands = customOp.getNonCtrlWireOperands();
    ASSERT_TRUE(nonCtrlWireOperands.size() == 1 && nonCtrlWireOperands[0] == args[0]);

    ValueRange ctrlWireOperands = customOp.getCtrlWireOperands();
    ASSERT_TRUE(ctrlWireOperands.size() == 1 && ctrlWireOperands[0] == args[1]);

    ValueRange ctrlValueOperands = customOp.getCtrlValueOperands();
    ASSERT_TRUE(ctrlValueOperands.size() == 1 && ctrlValueOperands[0] == args[3]);

    ASSERT_TRUE(customOp.getAdjointFlag());

    ValueRange allParams = customOp.getAllParams();
    ASSERT_TRUE(allParams.size() == 2 && allParams[0] == args[2] && allParams[1] == args[2]);

    ASSERT_TRUE(customOp.getParam(0) == args[2]);
    ASSERT_TRUE(customOp.getParam(1) == args[2]);
}

TEST(InterfaceTests, setWireOperands)
{
    std::string moduleStr = R"mlir(
func.func @f(%w0: i64, %w1: i64, %bool: i1) {
    ref_quantum.custom "Rot"() %w0 ctrls (%w1) ctrlvals (%bool) : i64 ctrls i64
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::ref_quantum::RefQuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::ref_quantum::CustomOp customOp = *f.getOps<catalyst::ref_quantum::CustomOp>().begin();

    Block &bb = f.getCallableRegion()->front();
    auto args = bb.getArguments();

    // Run checks
    customOp.setWireOperands({args[1], args[0]});
    std::vector<Value> wireOperands = customOp.getWireOperands();
    ASSERT_TRUE(wireOperands.size() == 2 && wireOperands[0] == args[1] &&
                wireOperands[1] == args[0]);
}

TEST(InterfaceTests, setNonCtrlWireOperands)
{
    std::string moduleStr = R"mlir(
func.func @f(%w0: i64, %w1: i64, %w2: i64, %bool: i1) {
    ref_quantum.custom "Rot"() %w0 ctrls (%w1) ctrlvals (%bool) : i64 ctrls i64
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::ref_quantum::RefQuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::ref_quantum::CustomOp customOp = *f.getOps<catalyst::ref_quantum::CustomOp>().begin();

    Block &bb = f.getCallableRegion()->front();
    auto args = bb.getArguments();

    // Run checks
    customOp.setNonCtrlWireOperands({args[2]});
    ValueRange nonCtrlWireOperands = customOp.getNonCtrlWireOperands();
    ASSERT_TRUE(nonCtrlWireOperands.size() == 1 && nonCtrlWireOperands[0] == args[2]);
}

TEST(InterfaceTests, setCtrlWireOperands)
{
    std::string moduleStr = R"mlir(
func.func @f(%w0: i64, %w1: i64, %w2: i64, %bool: i1) {
    ref_quantum.custom "Rot"() %w0 ctrls (%w1) ctrlvals (%bool) : i64 ctrls i64
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::ref_quantum::RefQuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::ref_quantum::CustomOp customOp = *f.getOps<catalyst::ref_quantum::CustomOp>().begin();

    Block &bb = f.getCallableRegion()->front();
    auto args = bb.getArguments();

    // Run checks
    customOp.setCtrlWireOperands({args[2]});
    ValueRange ctrlWireOperands = customOp.getCtrlWireOperands();
    ASSERT_TRUE(ctrlWireOperands.size() == 1 && ctrlWireOperands[0] == args[2]);
}

TEST(InterfaceTests, setCtrlValueOperands)
{
    std::string moduleStr = R"mlir(
func.func @f(%w0: i64, %w1: i64, %bool: i1, %other_bool: i1) {
    ref_quantum.custom "Rot"() %w0 ctrls (%w1) ctrlvals (%bool) : i64 ctrls i64
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::ref_quantum::RefQuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::ref_quantum::CustomOp customOp = *f.getOps<catalyst::ref_quantum::CustomOp>().begin();

    Block &bb = f.getCallableRegion()->front();
    auto args = bb.getArguments();

    // Run checks
    customOp.setCtrlValueOperands({args[3]});
    ValueRange ctrlValueOperands = customOp.getCtrlValueOperands();
    ASSERT_TRUE(ctrlValueOperands.size() == 1 && ctrlValueOperands[0] == args[3]);
}

TEST(InterfaceTests, setAdjointFlag)
{
    std::string moduleStr = R"mlir(
func.func @f(%w0: i64) {
    ref_quantum.custom "PauliX"() %w0 : i64
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::ref_quantum::RefQuantumDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::ref_quantum::CustomOp customOp = *f.getOps<catalyst::ref_quantum::CustomOp>().begin();

    // Run checks
    customOp.setAdjointFlag(true);
    ASSERT_TRUE(customOp.getAdjointFlag());

    customOp.setAdjointFlag(false);
    ASSERT_TRUE(!customOp.getAdjointFlag());
}

} // namespace
