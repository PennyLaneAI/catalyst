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

#include "QRef/IR/QRefInterfaces.h"
#include "QRef/IR/QRefOps.h"

using namespace mlir;

namespace {

TEST(InterfaceTests, Getters)
{
    std::string moduleStr = R"mlir(
func.func @f(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %param: f64, %bool: i1) {
    qref.custom "Rot"(%param, %param) %q0 adj ctrls (%q1) ctrlvals (%bool) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::qref::QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::qref::CustomOp customOp = *f.getOps<catalyst::qref::CustomOp>().begin();

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
func.func @f(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %bool: i1) {
    qref.custom "Rot"() %q0 ctrls (%q1) ctrlvals (%bool) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::qref::QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::qref::CustomOp customOp = *f.getOps<catalyst::qref::CustomOp>().begin();

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
func.func @f(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %q2: !qref.qubit_ref, %bool: i1) {
    qref.custom "Rot"() %q0 ctrls (%q1) ctrlvals (%bool) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::qref::QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::qref::CustomOp customOp = *f.getOps<catalyst::qref::CustomOp>().begin();

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
func.func @f(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %q2: !qref.qubit_ref, %bool: i1) {
    qref.custom "Rot"() %q0 ctrls (%q1) ctrlvals (%bool) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::qref::QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::qref::CustomOp customOp = *f.getOps<catalyst::qref::CustomOp>().begin();

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
func.func @f(%q0: !qref.qubit_ref, %q1: !qref.qubit_ref, %bool: i1, %other_bool: i1) {
    qref.custom "Rot"() %q0 ctrls (%q1) ctrlvals (%bool) : !qref.qubit_ref ctrls !qref.qubit_ref
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::qref::QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::qref::CustomOp customOp = *f.getOps<catalyst::qref::CustomOp>().begin();

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
func.func @f(%q0: !qref.qubit_ref) {
    qref.custom "PauliX"() %q0 : !qref.qubit_ref
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::qref::QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::qref::CustomOp customOp = *f.getOps<catalyst::qref::CustomOp>().begin();

    // Run checks
    customOp.setAdjointFlag(true);
    ASSERT_TRUE(customOp.getAdjointFlag());

    customOp.setAdjointFlag(false);
    ASSERT_TRUE(!customOp.getAdjointFlag());
}

TEST(InterfaceTests, globalPhase)
{
    std::string moduleStr = R"mlir(
func.func @f(%q0: !qref.qubit_ref, %cv: i1, %param: f64) {
    qref.gphase(%param) adj ctrls (%q0) ctrlvals (%cv) : f64 ctrls !qref.qubit_ref
    return
}
  )mlir";

    // Parsing boilerplate
    DialectRegistry registry;
    registry.insert<func::FuncDialect, catalyst::qref::QRefDialect>();
    MLIRContext context(registry);
    ParserConfig config(&context, /*verifyAfterParse=*/false);
    OwningOpRef<ModuleOp> mod = parseSourceString<ModuleOp>(moduleStr, config);

    // Parse ops
    func::FuncOp f = *(*mod).getOps<func::FuncOp>().begin();
    catalyst::qref::GlobalPhaseOp gphaseOp = *f.getOps<catalyst::qref::GlobalPhaseOp>().begin();

    Block &bb = f.getCallableRegion()->front();
    auto args = bb.getArguments();

    // Run checks
    ValueRange allParams = gphaseOp.getAllParams();
    ASSERT_TRUE(allParams.size() == 1 && allParams[0] == args[2]);

    ASSERT_TRUE(gphaseOp.getAdjointFlag());

    ValueRange ctrlWireOperands = gphaseOp.getCtrlWireOperands();
    ASSERT_TRUE(ctrlWireOperands.size() == 1 && ctrlWireOperands[0] == args[0]);

    ValueRange ctrlValueOperands = gphaseOp.getCtrlValueOperands();
    ASSERT_TRUE(ctrlValueOperands.size() == 1 && ctrlValueOperands[0] == args[1]);
}

} // namespace
