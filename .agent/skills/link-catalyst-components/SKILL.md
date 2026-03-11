---
name: link-catalyst-components
description: Use when configuring the build system (CMake) for new Catalyst components or linking against MLIR/LLVM.
---

# Link Catalyst Components

## Overview
This skill provides guidance on modifying `CMakeLists.txt` files to correctly link new C++ components against the Catalyst runtime and MLIR libraries.

## When to Use
- Adding a new translation pass (e.g., `TranslateToQASM3.cpp`).
- Creating a new standalone tool that uses Catalyst libraries.
- Fixing "undefined symbol" errors during linking.

## Key Concepts
1.  **MLIR Libraries**: Use `mlir_tablegen` and `add_mlir_translation_library`.
2.  **Catalyst Dialects**: Link against `QuantumDialect`, `GradientDialect`, etc.
3.  **LLVM Components**: Use `llvm_map_components_to_libnames` if needing core LLVM features (Support, Core).

## Implementation Pattern

### 1. Define the Library
In your `CMakeLists.txt`:

```cmake
add_mlir_translation_library(
  MlirToQasm3Translation
  TranslateToQASM3.cpp
  
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRQuantumDialect # Catalyst dialect
)
```

### 2. Locate MLIR/Catalyst
Ensure your project can find the Catalyst build artifacts. If building *within* the Catalyst tree (e.g., in `mlir/lib/Target/`), this is automatic. If building *outside*, you need `find_package(MLIR REQUIRED CONFIG)` and `find_package(Catalyst REQUIRED)`.

### 3. Registering the Translation
If creating a standalone tool (like `catalyst-translate`), link your library and register it:

```cmake
add_llvm_tool(catalyst-translate
  catalyst-translate.cpp
  DEPENDS
  MlirToQasm3Translation
)

target_link_libraries(catalyst-translate
  PRIVATE
  MlirToQasm3Translation
  MLIRSupport
  # ... other dialects
)
```

## Common Issues
- **Missing Dialect Registration**: Even if linked, you must register the dialect in C++ code (`registry.insert<...>`).
- **Visibility**: Use `PUBLIC` linking for libraries that expose MLIR types in their headers.
- **TableGen**: If your pass uses custom TableGen definitions, ensure `mlir_tablegen` is run and dependencies are set.
