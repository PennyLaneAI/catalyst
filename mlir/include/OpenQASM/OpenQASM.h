#pragma once

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// Dialect header

#include "OpenQASM/OpenQASMDialect.h.inc"

// Types header

#define GET_TYPEDEF_CLASSES
#include "OpenQASM/OpenQASMTypes.h.inc"

// Operations header

#define GET_OP_CLASSES
#include "OpenQASM/OpenQASM.h.inc"
