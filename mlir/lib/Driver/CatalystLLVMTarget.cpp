// Copyright 2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include "Driver/CatalystLLVMTarget.h"
#include "Gradient/IR/GradientDialect.h"

using namespace mlir;

void catalyst::driver::registerLLVMTranslations(DialectRegistry &registry)
{
    registerLLVMDialectTranslation(registry);
    registerBuiltinDialectTranslation(registry);
}

LogicalResult catalyst::driver::compileObjectFile(const CompilerOptions &options,
                                                  std::shared_ptr<llvm::Module> llvmModule,
                                                  llvm::TargetMachine *targetMachine,
                                                  StringRef filename)
{
    using namespace llvm;

    std::error_code errCode;
    raw_fd_ostream dest(filename, errCode, sys::fs::OF_None);

    if (errCode) {
        CO_MSG(options, Verbosity::Urgent, "could not open file: " << errCode.message() << "\n");
        return failure();
    }

    legacy::PassManager pm;
    if (targetMachine->addPassesToEmitFile(pm, dest, nullptr, CodeGenFileType::ObjectFile)) {
        CO_MSG(options, Verbosity::Urgent, "TargetMachine can't emit an .o file\n");
        return failure();
    }

    pm.run(*llvmModule);
    dest.flush();
    return success();
}
