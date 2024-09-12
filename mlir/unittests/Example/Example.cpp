#include "mlir/Parser/Parser.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Quantum/IR/QuantumOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace mlir;

namespace {
TEST(MLIRParser, ParseQuantumIR) {
  std::string moduleStr = R"mlir(
func.func @test_alloc_dealloc_no_fold() -> !quantum.bit {
  %r = quantum.alloc(3) : !quantum.reg
  %q = quantum.extract %r[0] : !quantum.reg -> !quantum.bit
  quantum.dealloc %r : !quantum.reg
  return %q : !quantum.bit
}
  )mlir";

  DialectRegistry registry;
  registry.insert<func::FuncDialect, catalyst::quantum::QuantumDialect>();
  MLIRContext context(registry);

  ParserConfig config(&context, /*verifyAfterParse=*/false);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(moduleStr, config);
  ASSERT_TRUE(module);
}
}
