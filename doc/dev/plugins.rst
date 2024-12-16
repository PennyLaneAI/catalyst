MLIR Plugins
============

This page outlines documentation on how to start developping an MLIR plugin that can work with Catalyst.
An MLIR plugin is a shared object that implements a compilation pass compatible with the MLIR framework.
Catalyst is built on top of MLIR, this means that MLIR plugins work with Catalyst.
This can enable anyone to build quantum compilation passes and new dialects as well.
So, let's get started!

Building the Standalone Plugin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Catalyst comes with ``Makefile`` rules to build the `standalone-plugin from MLIR upstream's source code <https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone>`_.
Simply type 

``make standalone-plugin``

and in the ``catalyst/mlir/standalone/build/lib`` folder, you will find the ``StandalonePlugin.so`` plugin.
The ``StandalonePlugin.so`` file is a simple plugin that has its own dialect (called Standalone dialect) and a single transformation that transforms symbol names from ``bar`` to ``foo``.
It is intended to show how one would build an MLIR plugin, rather than showing all the features to build a usable MLIR plugin.

You can use the ``StandalonePlugin.so`` plugin

* with either ``quantum-opt`` or ``catalyst-cli``,
* load it from Python and transform a quantum program.

For example, if you are interested in using it from the command line interface, you can use the following flags to load the standalone plugin:

* ``--load-pass-plugin=/path/to/StandalonePlugin.so``
* ``--load-dialect-plugin=/path/to/StandalonePlugin.so``

This allows all normal flags to work.
For example using ``quantum-opt --help`` while loading your pass plugin will enable you to see the documentation available for the standalone pass.

.. code-block::

    --standalone-switch-bar-foo                            -   Switches the name of a FuncOp named `bar` to `foo` and folds.

Taking into account the description of the pass ``standalone-switch-bar-foo``, let's write the most minimal program that would be transformed by this transformation.

.. code-block:: mlir

    module @module {
      func.func private @bar() -> (tensor<i64>) {
        %c = stablehlo.constant dense<0> : tensor<i64>
        return %c : tensor<i64>
      }
    }

And you can schedule this pass as any other pass 

.. code-block::

    quantum-opt --load-pass-plugin=/path/to/StandalonePlugin.so --pass-pipeline='builtin.module(standalone-switch-bar-to-foo) example.mlir'

And you have your transformed program

.. code-block:: mlir

    module @module {
      func.func private @foo() -> tensor<i64> {
        %c = stablehlo.constant dense<0> : tensor<i64>
        return %c : tensor<i64>
      }
    }

Notice that the name of the function ``bar`` has been changed to ``foo``.

Pass Plugins vs Dialect Plugins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may now be asking, "how come we used the option ``--load-pass-plugin`` but we didn't use the option ``--load-dialect-plugin``?"
The ``--load-pass-plugin`` option is used to load passes, while the ``--load-dialect-plugin`` is used to load dialects.
As mentioned earlier, the ``StandalonePlugin.so`` file also contains a dialect.
It is a simple dialect intended only for testing purposes, and it only contains a single operation. It is the ``standalone.foo`` operation.
(Please do not confuse this operation with symbols named ``foo``).

We can write a program that contains operations in the standalone dialect:

.. code-block:: mlir

    module @module {
      func.func private @bar() -> (i32) {
        %0 = arith.constant 0 : i32
        %1 = standalone.foo %0 : i32
        return %1 : i32
      }
    }

But if we try to run it, using the same command as shown earlier 

.. code-block::

      quantum-opt --load-pass-plugin=/path/to/StandalonePlugin.so --pass-pipeline='builtin.module(standalone-switch-bar-to-foo) example.mlir'

the compilation will fail with a message similar to:

.. code-block::

    example.mlir:4:10: error: Dialect `standalone' not found for custom op 'standalone.foo' 
    %1 = standalone.foo %0 : i32
         ^
    a.mlir:4:10: note: Registered dialects: acc, affine, amdgpu, amx, arith, arm_neon, arm_sme, arm_sve, async, bufferization, builtin, catalyst, cf, chlo, complex, dlti, emitc, func, gpu, gradient, index, irdl, linalg, llvm, math, memref, mesh, mhlo, mitigation, ml_program, mpi, nvgpu, nvvm, omp, pdl, pdl_interp, polynomial, quant, quantum, rocdl, scf, shape, sparse_tensor, spirv, stablehlo, tensor, test, tosa, transform, ub, vector, vhlo, x86vector, xegpu ; for more info on dialect registration see https://mlir.llvm.org/getting_started/Faq/#registered-loaded-dependent-whats-up-with-dialects-management

to be able to parse this dialect, we need to load the dialect which is stored in the same file

.. code-block::

    quantum-opt --load-pass-plugin=/path/to/StandalonePlugin.so --load-dialect-plugin-/path/to/StandalonePlugin.so --pass-pipeline='builtin.module(standalone-switch-bar-to-foo) example.mlir'

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

.. code-block:: diff

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

You will also need to make the following change:

.. code-block:: diff

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

.. code-block:: diff

    diff --git a/mlir/standalone/CMakeLists.txt b/mlir/standalone/CMakeLists.txt
    index 280cd80e1..fd6ee8f10 100644
    --- a/mlir/standalone/CMakeLists.txt
    +++ b/mlir/standalone/CMakeLists.txt
    @@ -32,8 +32,8 @@ if(MLIR_ENABLE_BINDINGS_PYTHON)
       mlir_configure_python_dev_packages()
     endif()
     
    -set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
    -set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR})
    +set(STANDALONE_SOURCE_DIR ${PROJECT_SOURCE_DIR}/standalone)
    +set(STANDALONE_BINARY_DIR ${PROJECT_BINARY_DIR}/standalone)
     include_directories(${LLVM_INCLUDE_DIRS})
     include_directories(${MLIR_INCLUDE_DIRS})
     include_directories(${STANDALONE_SOURCE_DIR}/include)

With these changes, you should now be able to use ``make all`` and build the standalone plugin.
Please note that the location of the ``StandalonePlugin.so`` shared object has changed.
It will now be stored in the ``mlir/build/lib/`` folder.

2. Include the header files in the standalone plugin pass.

.. code-block:: diff

    diff --git a/mlir/standalone/lib/Standalone/StandalonePasses.cpp b/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    index a23d0420f..83e2ce255 100644
    --- a/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    +++ b/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    @@ -12,6 +12,7 @@
     #include "mlir/Transforms/GreedyPatternRewriteDriver.h"
     
     #include "Standalone/StandalonePasses.h"
    +#include "Quantum/IR/QuantumOps.h"
     
     namespace mlir::standalone {
     #define GEN_PASS_DEF_STANDALONESWITCHBARFOO

You can type ``make all`` and see the compilation succeed.
Please note that Catalyst has three custom dialects, the Quantum, Catalyst and Gradient dialect.
Depending on which dialect you are interested in, you can include the definition of the operations in that way.

3. Marking dialects as dependent in the pass TableGen file.

.. code-block:: diff

    diff --git a/mlir/standalone/include/Standalone/StandalonePasses.td b/mlir/standalone/include/Standalone/StandalonePasses.td
    index dc8fb43d2..29510d74d 100644
    --- a/mlir/standalone/include/Standalone/StandalonePasses.td
    +++ b/mlir/standalone/include/Standalone/StandalonePasses.td
    @@ -26,6 +26,10 @@ def StandaloneSwitchBarFoo: Pass<"standalone-switch-bar-foo", "::mlir::ModuleOp"
         ```
       }];
     
    +   let dependentDialects = [
    +       "catalyst::quantum::QuantumDialect"
    +   ];
    +
     }
     
     #endif // STANDALONE_PASS

LLVM and MLIR use an embedded DSL to declare passes called `Tablegen <https://llvm.org/docs/TableGen/>`_.
This saves LLVM and MLIR developers time, because Tablegen generates C++ files that are mostly just boilerplate code.
We are not going to go in depth into Tablegen, you just need to know that transformations require to register which passes are used.
In this example, since we are interested in using the quantum dialect, we will add the Quantum Dialect in the list of dependent dialects.

One also needs to link the MLIRQuantum library and change the plugin tool to catalyst-cli.

.. code-block:: diff

    diff --git a/mlir/standalone/lib/Standalone/CMakeLists.txt b/mlir/standalone/lib/Standalone/CMakeLists.txt
    index 0f1705a25..8874e410d 100644
    --- a/mlir/standalone/lib/Standalone/CMakeLists.txt
    +++ b/mlir/standalone/lib/Standalone/CMakeLists.txt
    @@ -10,9 +10,11 @@ add_mlir_dialect_library(MLIRStandalone
             DEPENDS
             MLIRStandaloneOpsIncGen
             MLIRStandalonePassesIncGen
    +        MLIRQuantum
     
             LINK_LIBS PUBLIC
             MLIRIR
             MLIRInferTypeOpInterface
             MLIRFuncDialect
    +        MLIRQuantum
             )

.. code-block:: diff

    diff --git a/mlir/standalone/standalone-plugin/CMakeLists.txt b/mlir/standalone/standalone-plugin/CMakeLists.txt
    index 3e3383608..2dbeea9d5 100644
    --- a/mlir/standalone/standalone-plugin/CMakeLists.txt
    +++ b/mlir/standalone/standalone-plugin/CMakeLists.txt
    @@ -5,7 +5,7 @@ add_llvm_library(StandalonePlugin
             DEPENDS
             MLIRStandalone
             PLUGIN_TOOL
    -        mlir-opt
    +        catalyst-cli
     
             LINK_LIBS
             MLIRStandalone

Please note that if you are using the Catalyst or Gradient dialects, you should also add MLIRCatalyst and MLIRGradient to the list of dependences and libraries to be linked.

4. Modify the standalone plugin to modify quantum operations.

Here we will create a very simple pass that will change a the quantum qubit allocation from 1 to 42.
Yes, this is also a very simple and unnecessary task, but just one to illustrate a little bit how MLIR works.
We recommend reading MLIR tutorials on how to write MLIR passes, reading the Catalyst source to understand the Catalyst IR, and submitting issues if you are having troubles building your own plugin.

The first thing we need to do is change the ``OpRewritePattern`` to match against our ``quantum::AllocOp`` which denotes how many qubits should be allocated for a given quantum program.

.. code-block:: diff

    diff --git a/mlir/standalone/lib/Standalone/StandalonePasses.cpp b/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    index 83e2ce255..504cf2d20 100644
    --- a/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    +++ b/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    @@ -19,10 +19,10 @@ namespace mlir::standalone {
     #include "Standalone/StandalonePasses.h.inc"
     
     namespace {
    -class StandaloneSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
    +class StandaloneSwitchBarFooRewriter : public OpRewritePattern<catalyst::quantum::AllocOp> {
     public:
    -  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
    -  LogicalResult matchAndRewrite(func::FuncOp op,
    +  using OpRewritePattern<catalyst::quantum::AllocOp>::OpRewritePattern;
    +  LogicalResult matchAndRewrite(catalyst::quantum::AllocOp op,
                                     PatternRewriter &rewriter) const final {
         if (op.getSymName() == "bar") {
           rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });

The next step is changing the contents of the function itself:

.. code-block:: diff

    diff --git a/mlir/standalone/lib/Standalone/StandalonePasses.cpp b/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    index 83e2ce255..e8a7f805e 100644
    --- a/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    +++ b/mlir/standalone/lib/Standalone/StandalonePasses.cpp
    @@ -19,15 +19,21 @@ namespace mlir::standalone {
     #include "Standalone/StandalonePasses.h.inc"
     
     namespace {
    -class StandaloneSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
    +class StandaloneSwitchBarFooRewriter : public OpRewritePattern<catalyst::quantum::AllocOp> {
     public:
    -  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
    -  LogicalResult matchAndRewrite(func::FuncOp op,
    +  using OpRewritePattern<catalyst::quantum::AllocOp>::OpRewritePattern;
    +  LogicalResult matchAndRewrite(catalyst::quantum::AllocOp op,
                                     PatternRewriter &rewriter) const final {
    -    if (op.getSymName() == "bar") {
    -      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
    +    // get the number of qubits allocated
    +    if (op.getNqubitsAttr().value_or(0) == 1) {
    +      Type i64 = rewriter.getI64Type();
    +      auto fortytwo = rewriter.getIntegerAttr(i64, 42);
    +
    +      // modify the allocation to change the number of qubits to 42.
    +      rewriter.modifyOpInPlace(op, [&]() { op.setNqubitsAttrAttr(fortytwo); });
           return success();
         }
    +    // failure indicates that nothing was modified.
         return failure();
       }
     };

And then we can run ``make all`` again.
The shared object of the standalone plugin should be available in ``mlir/build/lib/StandalonePlugin.so``.
This shared object can be used with ``catalyst-cli`` and ``quantum-opt``
You can change the name of the pass, change the name of the shared object and make any changes you want to get started with your quantum compilation journey.
This was just an easy example to get started.

With the steps above, you can take an MLIR program with a ``quantum.alloc`` instruction which allocates statically 1 qubit, and the program will be transformed to allocate 42 qubits statically.

5. Build your own python wheel and ship your plugin.

Now that you have your ``StandalonePlugin.so``, you can ship it in a python wheel.
To allow users to run your pass, we have provided a class called :class:`~.passes.Pass` and :class:`~.passes.PassPlugin`.
You can extend these classes and allow the user to import your derived classes and run passes as a decorator.
For example:

.. code-block:: python

    @SwitchBarToFoo
    @qml.qnode(qml.device("lightning.qubit", wires=1))
    def qnode():
        return qml.state()

    @qml.qjit
    def module():
        return qnode()

If you inspect the MLIR sources, you'll find that the number of qubits allocated will be 42.
Take a look into the ``standalone_plugin_wheel`` make rule to see how we test shipping a plugin.
