MLIR Plugins
============

This page outlines documentation on how to start developping an MLIR plugin that can work with Catalyst.
An MLIR plugin is a shared object that implements a compilation pass compatible with the MLIR framework.
Catalyst is built on top of MLIR, this means that MLIR plugins work with Catalyst.
This can enable anyone to build quantum compilation passes and new dialects as well.
So, let's get started!

Building the Standalone Plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Catalyst comes with ``Makefile`` rules to build the standalone-plugin from MLIR upstream.
Simply type:

```
make standalone-plugin
```

and in the ``catalyst/mlir/standalone/build/lib`` folder, you will find the ``StandalonePlugin.so`` plugin.
The ``StandalonePlugin.so`` is a simple plugin that has its own dialect (called Standalone dialect) and a single transformation that transforms symbols from ``bar`` to ``foo``.
It is intended to show how one would build an MLIR plugin, rather than showing all the features to build a usable MLIR plugin.

With the ``StandalonePlugin.so`` plugin built, you can:

* use it on the command line with either ``quantum-opt`` or ``catalyst-cli``.
* load it from Python and schedule it in one of your quantum programs.

For example, if you are interested in using it from the command line interface, you can use the following flags to load the standalone plugin:

* ``--load-pass-plugin=/path/to/StandalonePlugin.so``
* ``--load-dialect-plugin=/path/to/StandalonePlugin.so``

This allows all normal flags to work.
For example using ``quantum-opt --help``, while loading your pass plugin will enable you to see the documentation available for the standalone pass.

```
      --standalone-switch-bar-foo                            -   Switches the name of a FuncOp named `bar` to `foo` and folds.
```

To run a pass you need a program to transform.
Taking into account the description of the pass ``standalone-switch-bar-foo``, let's write the most minimal program that would be transformed by this transformation.

```

module @module {
  func.func private @bar() -> (tensor<i64>) {
    %c = stablehlo.constant dense<0> : tensor<i64>
    return %c : tensor<i64>
  }
}

```

And you can schedule this pass as any other pass 

```
      quantum-opt --load-pass-plugin=/path/to/StandalonePlugin.so --pass-pipeline='builtin.module(standalone-switch-bar-to-foo) example.mlir'
```

And you have your transformed program

```

module @module {
  func.func private @foo() -> tensor<i64> {
    %c = stablehlo.constant dense<0> : tensor<i64>
    return %c : tensor<i64>
  }
}

```

Notice that the name of the function ``bar`` has been changed to ``foo``.

Pass Plugins vs Dialect Plugins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may now be asking, "how come we used the option ``--load-pass-plugin`` but we didn't use the option ``--load-dialect-plugin``?"
The ``--load-pass-plugin`` option is used to load passes, while the ``--load-dialect-plugin`` is used to load dialects.
As mentioned earlier, the ``StandalonePlugin.so`` file also contains a dialect.
It is a simple dialect intended only for learning purposes, and it only contains a single operation. It is the ``standalone.foo`` operation.
(Please do not confuse this operation with symbols named ``foo``).

We can write a program that contains operations in the standalone dialect:

```
module @module {
  func.func private @bar() -> (i32) {
    %0 = arith.constant 0 : i32
    %1 = standalone.foo %0 : i32
    return %1 : i32
  }
}
```

But if we try to run it, using the same command as shown earlier 

```
      quantum-opt --load-pass-plugin=/path/to/StandalonePlugin.so --pass-pipeline='builtin.module(standalone-switch-bar-to-foo) example.mlir'
```

the compilation will fail with a message similar to:

```
    example.mlir:4:10: error: Dialect `standalone' not found for custom op 'standalone.foo' 
    %1 = standalone.foo %0 : i32
         ^
a.mlir:4:10: note: Registered dialects: acc, affine, amdgpu, amx, arith, arm_neon, arm_sme, arm_sve, async, bufferization, builtin, catalyst, cf, chlo, complex, dlti, emitc, func, gpu, gradient, index, irdl, linalg, llvm, math, memref, mesh, mhlo, mitigation, ml_program, mpi, nvgpu, nvvm, omp, pdl, pdl_interp, polynomial, quant, quantum, rocdl, scf, shape, sparse_tensor, spirv, stablehlo, tensor, test, tosa, transform, ub, vector, vhlo, x86vector, xegpu ; for more info on dialect registration see https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management

```

to be able to parse this dialect, we need to load the dialect which is stored in the same file

```
      quantum-opt --load-pass-plugin=/path/to/StandalonePlugin.so --load-dialect-plugin-/path/to/StandalonePlugin.so --pass-pipeline='builtin.module(standalone-switch-bar-to-foo) example.mlir'
```

Now, you can parse the program without the error.

Creating your own Pass Plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Catalyst lists LLVM as a git submodule in its repository and the LLVM project already contains an example standalone plugin.
When running ``make standalone-plugin`` Catalyst will copy the directory containing the standalone plugin and patch it to make sure that it works with Catalyst.
However, as mentioned earlier, the standalone plugin is a bare bones example.
You may be wondering, well, how can I make a standalone plugin but that is able to change some aspects of the quantum program?
For that, you will need to change the build script for the standalone plugin.
For now, we found that the following process is the easiest one:

1. Add the standalone plugin directory as a subdirectory of Catalyst:

```
diff --git a/mlir/CMakeLists.txt b/mlir/CMakeLists.txt
index c0b8dfd6c..1b5c2e528 100644
--- a/mlir/CMakeLists.txt
+++ b/mlir/CMakeLists.txt
@@ -73,6 +73,7 @@ add_subdirectory(include)
 add_subdirectory(lib)
 add_subdirectory(tools)
 add_subdirectory(test)
+add_subdirectory(standalone)
 
 if(QUANTUM_ENABLE_BINDINGS_PYTHON)
   message(STATUS "Enabling Python API")
```

You will also need to make the following change:

```
diff --git a/mlir/standalone/CMakeLists.txt b/mlir/standalone/CMakeLists.txt
index e999ae34d..fd6ee8f10 100644
--- a/mlir/standalone/CMakeLists.txt
+++ b/mlir/standalone/CMakeLists.txt
@@ -1,6 +1,3 @@
-cmake_minimum_required(VERSION 3.20.0)
-project(standalone-dialect LANGUAGES CXX C)
-
 set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
 
 set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
```

2. Include the header files 
