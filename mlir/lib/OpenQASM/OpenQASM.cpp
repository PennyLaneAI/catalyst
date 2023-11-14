#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "OpenQASM/OpenQASM.h"

using namespace mlir;
using namespace catalyst::openqasm;

// Dialect source

#include "OpenQASM/OpenQASMDialect.cpp.inc"

void OpenQASMDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "OpenQASM/OpenQASMTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "OpenQASM/OpenQASM.cpp.inc"
        >();
}

// Types source

#define GET_TYPEDEF_CLASSES
#include "OpenQASM/OpenQASMTypes.cpp.inc"

// Operations source

#define GET_OP_CLASSES
#include "OpenQASM/OpenQASM.cpp.inc"
