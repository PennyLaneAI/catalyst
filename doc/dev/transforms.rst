Catalyst Compiler Passes
########################

..
    TODO: add MLIR syntax highlighting to these snippets
    TODO: add an end-to-end guide which includes compiling and using the custom pass

The MLIR Transformation Framework
=================================

Catalyst program transformations are implemented using the MLIR framework. A good starting point
is to understand the IR structure and how MLIR transformations are written. Refer to the resources
below for further information beyond this document.

- `The structure and elements of the MLIR program representation <https://mlir.llvm.org/docs/LangRef/>`_.

- `Understand the relationship of IR objects, and how they can be used to traverse the IR <https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/>`_.

- `Documentation on general MLIR transformations (or "passes") <https://mlir.llvm.org/docs/PassManagement/>`_.


In addition, many transformations can be expressed via pattern rewriting. A pattern is a specific
arrangement of a group of operations into a directed acyclic graph (DAG). This allows the framework
developer to provide a generic transformation mechanism, for which the compiler developer only needs
to provide source and replacement patterns in order to implement a transformation.
See the following resource for information on pattern rewrites:

- `Information on specific transformations called pattern rewrites <https://mlir.llvm.org/docs/PatternRewriter/>`_.


What does Catalyst's IR look like?
==================================

Let's look at a simple example starting in Python:

.. code-block:: python

    def circuit(x: complex):
        qml.Hadamard(wires=0)

        A = np.array([[1, 0], [0, np.exp(x)]])
        qml.QubitUnitary(A, wires=0)

        B = np.array([[1, 0], [0, np.exp(2*x)]])
        qml.QubitUnitary(B, wires=0)

        return measure(0)

The corresponding IR might look something like this (simplified):

.. code-block:: mlir

    func.func @circuit(%arg0: complex<f64>) -> i1 {
        %c00 = complex.constant [0.0, 0.0] : complex<f64>
        %c10 = complex.constant [1.0, 0.0] : complex<f64>
        %c20 = complex.constant [2.0, 0.0] : complex<f64>

        %0 = complex.exp %arg0 : complex<f64>
        %A = tensor.from_elements %c10, %c00, %c00, %0 : tensor<2x2xcomplex<f64>>

        %1 = complex.mul %arg0, %c20 : complex<f64>
        %2 = complex.exp %1 : complex<f64>
        %B = tensor.from_elements %c10, %c00, %c00, %2 : tensor<2x2xcomplex<f64>>

        %reg = quantum.alloc(1) : !quantum.reg
        %q0 = quantum.extract %reg[ 0] : !quantum.reg -> !quantum.bit
        %q1 = quantum.custom "Hadamard"() %q0 : !quantum.bit
        %q2 = quantum.unitary(%A : tensor<2x2xcomplex<f64>>) %q1 : !quantum.bit
        %q3 = quantum.unitary(%B : tensor<2x2xcomplex<f64>>) %q2 : !quantum.bit

        %m, %q4 = quantum.measure %q3 : i1, !quantum.bit

        return %m : i1
    }

A few things you might note in this representation:

- Qubits need to be extracted from the global register in each scope before they can be used by
  gates.

- Most quantum gates currently do not have an explicit operation defined for them, instead they
  share a common instruction ``quantum.custom`` and are identified by a string attribute.
  (This may change in the future.)

- Quantum gates not only receive qubits as arguments, but produce new qubit values as results. In
  this sense, the ``!quantum.bit`` data type can be thought of as the pseudo-state of a qubit at a
  particular point in the program.
  This allows the connectivity between gates on each qubit (the "circuit wires") to be explicit in
  the IR, via these data relations that MLIR calls *def-use* chains, linking the *usage* of a value
  (from an operation argument) to the *definition* of value (from an operation result).

- Classical instructions and quantum instructions are freely mixed in the IR, just like in the user
  source code.

The representation above is very handy for traversing quantum operations, in effect providing the
same capabilities of a directed acyclic graph (DAG) encoding of a quantum circuit. Meanwhile,
classical instructions as well as side-effecting operations (e.g., print) can also be present in the
same representation.


Writing and running your first Catalyst pass
============================================

If this is your first time writing MLIR or LLVM passes, the boilerplate can be quite overwhelming. 
Let's first set up the various boilerplate items required to register and run a new pass. 

We'll create an empty pass in the ``Catalyst`` dialect that just prints out hello world to stdout.
Note that the ``mlir/include`` (and ``mlir/lib``) directories consists of all the available dialects, so if you want to write a new pass in another dialect, it should be added to the subdirectory of that dialect. 

The first thing to do is to create the pass object in the tablegen ``mlir/include/Catalyst/Transforms/Passes.td``:

.. code-block::

    def MyHelloWorldPass : Pass<"my-hello-world"> {
        let summary = "An empty pass boilerplate that prints out hello world.";

        let constructor = "catalyst::createMyHelloWorldPass()";
    }

When the dialect is built, this tablegen def will be built to a C++ file ``mlir/build/include/Catalyst/Transforms/Passes.h.inc``, containing the newly defined object called ``MyHelloWorldPassBase``, alongside the various necessary boilerplate methods in the MLIR infrastructure. 
Tablegen is designed such that we don't have to write all that boilerplate ourselves. 

Now we write the pass itself. Create a new file ``mlir/lib/Catalyst/Transforms/MyHelloWorldPass.cpp`` with the following content:

.. code-block:: cpp

    #define DEBUG_TYPE "myhelloworld"

    #include "Catalyst/IR/CatalystDialect.h"
    #include "mlir/Pass/Pass.h"
    #include "llvm/Support/Debug.h"

    using namespace llvm;
    using namespace mlir;
    using namespace catalyst;

    namespace catalyst {
    #define GEN_PASS_DEF_MYHELLOWORLDPASS
    #define GEN_PASS_DECL_MYHELLOWORLDPASS
    #include "Catalyst/Transforms/Passes.h.inc"

    struct MyHelloWorldPass : public impl::MyHelloWorldPassBase<MyHelloWorldPass> {
        using impl::MyHelloWorldPassBase<MyHelloWorldPass>::MyHelloWorldPassBase;

        void runOnOperation() override { llvm::errs() << "Hello world!\n"; }
    };

    std::unique_ptr<Pass> createMyHelloWorldPass() { return std::make_unique<MyHelloWorldPass>(); }

    } // namespace catalyst

We make the pass object ``MyHelloWorldPass``, which inherits from the base class ``MyHelloWorldPassBase`` that tablegen will build in the namespace ``impl``. 
The function that determines what your pass actually does is the ``void runOnOperation()``. Here all the pass does is print out ``"Hello world!\n"``. 

(A sidenote on printing messages in MLIR: there are two major printing options in LLVM. The `more standard one <https://llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option>`_ is ``dbgs()``, which only prints when a debug flag is set. 
The other option is the ``errs()`` used here, which will print no matter what.)

This new C++ file needs to be added to the ``mlir/lib/Catalyst/Transforms/CMakeLists.txt`` file (or the CMakeLists.txt of whichever directory that has your new pass file): 

.. code-block::

    file(GLOB SRC
        ...
        MyHelloWorldPass.cpp
    )

After writing the pass, we need to register it in a few places. In ``mlir/include/Catalyst/Transforms/Passes.h``, add the method 

.. code-block:: cpp

    namespace catalyst {
        ...
        std::unique_ptr<mlir::Pass> createMyHelloWorldPass();
        ...
    }

And in ``mlir/lib/Catalyst/Transforms/RegisterAllPasses.cpp``, register the pass via 

.. code-block:: cpp

    void catalyst::registerAllCatalystPasses()
    {
        ...
        mlir::registerPass(catalyst::createMyHelloWorldPass);
        ...
    }

Note that this addition in ``RegisterAllPasses.cpp`` needs to happen in the ``lib/Catalyst/Transforms`` directory, regardless of which dialect your pass belongs to.

Now that we have written our shiny new pass, we can build it by going back to the top-level ``catalyst`` directory and 

.. code-block::

    make dialects

The tool to run passes is built as ``mlir/build/bin/quantum-opt``. 
Since this is an executable, it needs to be invoked as ``./quantum-opt`` instead of just plain ``quantum-opt`` (if you are in the ``mlir/build/bin`` directory; otherwise supply the full path).

We can inspect by all the available passes by running ``quantum-opt --help``:

.. code-block::

    OVERVIEW: Quantum optimizer driver
    ...
    USAGE: quantum-opt [options] <input file>

    OPTIONS:
        ...
        --my-hello-world                   -   An empty pass boilerplate that prints out hello world.

Here the displayed ``--help`` message will be the ``summary`` we wrote in the tablegen file. The command line option to run our new pass is the template string in the def line in the tablegen file. 

To run the pass, simply do 

.. code-block::

    ./mlir/build/bin/quantum-opt -my-hello-world input.mlir

on any input mlir file ``input.mlir``. And our new pass will print out ``Hello world!``. 

.. note::

    If you are encoutering issues, or would like to quickly try out the hello world pass described in this
    section, you can have a look at or cherry-pick this commit which includes all changes described
    in this section: https://github.com/PennyLaneAI/catalyst/commit/ba7b3438667963b307c07440acd6d7082f1960f3


Writing transformations on Catalyst's IR
========================================

We'll start with DAG-to-DAG transformations, which typically match small pieces of code at a time.
In our example above, we might to consider merging the two ``quantum.unitary`` applications because
they act on the same qubit in immediate succession:

.. code-block:: mlir

    %0 = complex.exp %arg0 : complex<f64>
    %A = tensor.from_elements %c10, %c00, %c00, %0 : tensor<2x2xcomplex<f64>>
    %q2 = quantum.unitary %A, %q1 : !quantum.bit                                (A)

    %1 = complex.mul %arg0, %c20 : complex<f64>
    %2 = complex.exp %1 : complex<f64>
    %B = tensor.from_elements %c10, %c00, %c00, %2 : tensor<2x2xcomplex<f64>>
    %q3 = quantum.unitary %B, %q2 : !quantum.bit                                (B)

Note how the value ``%q2`` links the two operations together from definition ``(A)`` to use ``(B)``
across several other instructions.

As seen in the `pattern rewriter documentation <https://mlir.llvm.org/docs/PatternRewriter/#defining-patterns>`_,
a new rewrite pattern can be defined as a C++ class as follows, where we will focus on the ``match``
and ``rewrite`` methods (refer to the link for the full class and up to date information):

.. code-block:: cpp

    struct QubitUnitaryFusion : public OpRewritePattern<QubitUnitaryOp>
    {
        ...

        LogicalResult match(QubitUnitaryOp op) const override {
            // The ``match`` method returns ``success()`` if the pattern is a match, failure
            // otherwise.
        }

        void rewrite(QubitUnitaryOp op, PatternRewriter &rewriter) {
            // The ``rewrite`` method performs mutations on the IR rooted at ``op`` using
            // the provided rewriter. All mutations must go through the provided rewriter.
        }

        ...
    };

Note that by inheriting from ``OpRewritePattern`` instead of the generic ``RewritePattern``,
operations will automatically be filtered and our pattern will only be invoked on
``QubitUnitaryOp`` objects.

The first step in pattern rewriting is the matching phase. We want to match the following pattern of
``QubitUnitary`` operations (represented in graph form, where the first argument is the matrix, and
the second is a list of qubits):

.. code-block::

    QubitUnitary(*, QubitUnitary(*, *))

Let's implement it in C++:

.. code-block:: cpp

    LogicalResult match(QubitUnitaryOp op) const override
    {
        ValueRange qbs = op.getInQubits();
        Operation *parent = qbs[0].getDefiningOp();

        // Parent should be a QubitUnitaryOp
        if (!isa<QubitUnitaryOp>(parent)) {
            return failure();
        }

        QubitUnitaryOp parentOp = cast<QubitUnitaryOp>(parent);
        ValueRange parentQbs = parentOp.getOutQubits();

        // Parent's output qubits should be the current op's input qubits,
        // and the qubits need to be in the same order
        if (qbs.size() != parentQbs.size()) {
            return failure();
        }

        for (auto [qb1, qb2] : llvm::zip(qbs, parentQbs))
            if (qb1 != qb2) {
                return failure();
            }

        return success();
    }

Note that we have used a couple of functions (like ``getInQubits`` and ``getOutQubits``) from the
definition of the ``QubitUnitaryOp`` class. Since we define our operations in the declarative ODS
(tablegen) format, the corresponding C++ classes are automatically generated. This is the definition
for the ``QubitUnitaryOp`` from the `QuantumOps.td <https://github.com/PennyLaneAI/catalyst/blob/201b0ec6cbec18b6411a876a3c72ba878123e2a1/mlir/include/Quantum/IR/QuantumOps.td#L267>`_
file:

.. code-block::

    def QubitUnitaryOp : Gate_Op<"unitary"> {
        let summary = "Apply an arbitrary fixed unitary matrix";
        let description = [{
            The `quantum.unitary` operation applies an arbitrary fixed unitary matrix to the
            state-vector. The arguments are a set of qubits and a 2-dim matrix of complex numbers
            that represents a Unitary matrix of size 2^(number of qubits) * 2^(number of qubits).
        }];

        let arguments = (ins
            2DTensorOf<[Complex<F64>]>:$matrix,
            Variadic<QubitType>:$in_qubits
        );

        let results = (outs
            Variadic<QubitType>:$out_qubits
        );

        let assemblyFormat = [{
            `(` $matrix `:` type($matrix) `)` $in_qubits attr-dict `:` type($out_qubits)
        }];
    }

MLIR will automatically generate canonical ``get*`` methods for attributes like ``in_qubits``,
``out_qubits``, and ``matrix``. When in doubt it's best to have a look at the generated C++ files in
the build folder, named ``QuantumOps.h.inc`` and ``QuantumOps.cpp.inc`` in this instance.

Alright, now that we have the matching part, let's implement the actual transformation via the
``rewrite`` method. All we need to do is replace the original pattern with the following:

.. code-block::

    QubitUnitary(A, QubitUnitary(B, Q))  -->  QubitUnitary(AxB, Q)

In C++ it will look as follows:

.. code-block:: cpp

    void rewrite(QubitUnitaryOp op, PatternRewriter &rewriter) const override
    {
        ValueRange qbs = op.getInQubits();
        QubitUnitaryOp parentOp = cast<QubitUnitaryOp>(qbs[0].getDefiningOp());

        // In the tablegen definition of `QubitUnitaryOp`, there is a
        // field called `$matrix`, storing the matrix for the unitary gate.
        // Tablegen automatically generates getters for all of the fields.
        mlir::Value m1 = op.getMatrix();
        mlir::Value m2 = parentOp.getMatrix();

        // Get the type of a 2x2 complex matrix
        // Note that both m1 and m2 have this type already
        mlir::Type MatrixType = m1.getType();

        // Create the matrix multiplication operation
        // The linalg.matmul op's semantics is:
        //   linalg.matmul({A, B}, {C})
        // performs C+=A*B
        // so we need to create a zero matrix of the desired type and shape first
        tensor::EmptyOp zeromat =
            rewriter.create<tensor::EmptyOp>(op.getLoc(), MatrixType, ValueRange{});

        // The first argument to the `create` need to be a `Location`
        // which can usually just be a `getLoc()` from any operation you have handy
        // The second argument needs to be (a list of) type(s) of the operation's output
        // The third argument needs to be (a list of) input value(s) to the operation
        linalg::MatmulOp matmul = rewriter.create<linalg::MatmulOp>(
            op.getLoc(), TypeRange{MatrixType}, ValueRange{m1, m2}, ValueRange{zeromat});

        // Some peculiarity for the matmul operation; no need to worry about it here
        matmul->setAttr("operandSegmentSizes", rewriter.getDenseI32ArrayAttr({2, 1}));

        // Replace the matrix for the parent unitary (which is the first unitary op)
        // with the product matrix
        // Note: we need to move the zero matrix
        // and the matmul before the parent unitary
        // so all of them are defined before being used by the parent unitary
        zeromat->moveBefore(parentOp);
        matmul->moveBefore(parentOp);
        mlir::Value res = matmul.getResult(0);
        parentOp->setOperand(0, res);

        // The second unitary is not needed anymore
        // Whoever uses the second unitary, use the first one instead!
        op.replaceAllUsesWith(parentOp);
    }

When writing transformations, the rewriter is the most important tool we have. It can create new
operations for us, delete others, or change the place in the IR where we are choosing to make
changes (also called the insertion point). Let's have look at some of these elements:

- **Constructing new operations**:

  New operations are created via the ``rewriter.create`` method. Here we want to generate a matrix
  multiplication instruction from the ``linalg`` dialect. C++ namespaces usually correspond to the
  dialect name. The first thing the rewriter needs is always a `location object <https://mlir.llvm.org/docs/Diagnostics/#source-locations>`_,
  which is used in debugging to refer back to the original source code line, for example.
  Following this, we need to provide the right arguments to instantiate the operation. So-called
  operation builders are automatically defined for this purpose, whose source can be referenced to
  consult which arguments are required. Looking into ``LinalgStructuredOps.h.inc`` for example
  reveals the following options:

  .. code-block:: cpp

    static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
    static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
    static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
    static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute cast, ArrayRef<NamedAttribute> attributes = {});

  We can always ignore the first two arguments, ``odsBuilder`` and ``odsState``, but the remaining
  ones are the arguments we'll need to provide to the rewriter. We chose the simplest one which
  only requires specifying a range of values for the operation ``inputs`` (two to be precise). We
  can ignore ``outputs`` argument for now as it is a peculiarity of the ``linalg`` dialect.
  If necessary, the result types of an operation may be specified as can be seen in the second
  version, but for ``matmul`` the result types can be automatically deduced.

- **Removing operations**:

  We can remove operations via the ``rewriter.replaceOp`` method (among others). The reason we
  don't straight up delete operations is that that would break the def-chains in the IR. Instead,
  we always need to provide replacement values for the results that the operation to be deleted
  defined.
  In this case, we simply replace the output qubit values with the input qubit values to maintain
  the correct "wire" connections. We would thus change

  .. code-block:: mlir

      %q2 = quantum.unitary %A, %q1 : !quantum.bit
      %q3 = quantum.unitary %B, %q2 : !quantum.bit

  into

  .. code-block:: mlir

      %q3 = quantum.unitary %B, %q1 : !quantum.bit

  Note how the argument of the second unitary op was automatically swapped from ``%q2`` to
  ``%q1``.

- **Updating operations**:

  Operation arguments and `attributes <https://mlir.llvm.org/docs/LangRef/#attributes>`_ can also
  be modified in-place (without creating a new operation). We use this to replace the matrix
  argument of our operation with the result of the multiplication. Since this mechanism doesn't
  go through the rewriter, he have to notify it explicitly that we are making changes to an
  operation:

  .. code-block:: cpp

        parentOp->setOperand(0, res);


Invoking transformation patterns
================================

IR changes are always effected by a transformation *pass*. Many compilers are structured around the
notion of passes, where the program is progressively transformed and each pass is responsible for a
particular sub-task.

While the transformation pattern we wrote above defines how we want to transform certain aspects of
our program, it doesn't yet specify how the patterns are applied to an input program. For this we
need to write a pass.

The simplest approach might be to say we want our transformation pass to look at the entire program,
and apply a set of patterns we defined like the one above. We can do so by creating an
`OperationPass <https://mlir.llvm.org/docs/PassManagement/#operation-pass-static-filtering-by-op-type>`_
that acts on an MLIR module (remember an MLIR module is an operation that itself contains globals
and other function operations, which themselves can contain other operations, and so on):

.. code-block:: cpp

    struct QuantumOptimizationPass : public PassWrapper<QuantumOptimizationPass, OperationPass<ModuleOp>>
    {
        void runOnOperation() {
            // Get the current operation being operated on.
            ModuleOp op = getOperation();
            MLIRContext *ctx = &getContext();

            // Define the set of patterns to use.
            RewritePatternSet quantumPatterns(ctx);
            quantumPatterns.add<QubitUnitaryFusion>(ctx);

            // Apply patterns in an iterative and greedy manner.
            if (failed(applyPatternsAndFoldGreedily(op, std::move(quantumPatterns)))) {
                return signalPassFailure();
            }
        }
    };

To apply patterns we need a `pattern applicator <https://mlir.llvm.org/docs/PatternRewriter/#common-pattern-drivers>`_.
There a few in MLIR but typically you can just use the greedy pattern rewrite driver
(``applyPatternsAndFoldGreedily``), which will iterative over the IR and apply patterns until a
fixed point is reached.

.. note::

    If you are encoutering issues, or would like to quickly try out the merge unitary pass described in this
    section, you can have a look at or cherry-pick this commit which includes all changes described
    in this section: https://github.com/PennyLaneAI/catalyst/commit/2e7f7cde8cf65091e0f77cb0ccf2c5762501ee11


Writing more general transformations
====================================

The pattern-based approach to transformations is not limited to small peephole optimizations like
the one above, in fact all transformation passes in Catalyst currently use either regular rewrite
patterns or dialect conversion patterns. Let's take a quick look at the finite-difference method
in Catalyst for example.

The starting point for the transformation is the differentiation instruction in our gradient dialect
(`GradOp <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/include/Gradient/IR/GradientOps.td#L25>`_).
It acts like a function call, but instead returns the derivative of the function for some given
inputs:

.. code-block:: mlir

    func.func @my_func(f64, f64, f64) -> f64 {
        ...
    }

    %deriv:3 = gradient.grad "fd" @my_func(%x, %y, %z) : (f64, f64, f64) -> (f64, f64, f64)

We'll want to replace this with code that implements the finite-difference method. The *pass*
implementation will essentially look like the one above (say ``GradientPass``), but with a different
pattern set. This pattern would instead act on all ``GradOp`` objects in the program:

.. code-block:: cpp

    struct FiniteDiffLowering : public OpRewritePattern<GradOp>

But since the gradient could be calculated in different ways, we want to filter matches to those
gradient ops that specify the finite-difference method, indicated via the ``"fd"``
`attribute <https://mlir.llvm.org/docs/LangRef/#attributes>`_:

.. code-block:: cpp

    LogicalResult FiniteDiffLowering::match(GradOp op)
    {
        if (op.getMethod() == "fd")
            return success();

        return failure();
    }

For the rewriting part we'll want to introduce a few new elements, such as looking up symbols
(function names), creating new functions, and changing the insertion point.

.. code-block:: cpp

    void FiniteDiffLowering::rewrite(GradOp op, PatternRewriter &rewriter)
    {
        // First let's find the function the grad operation is referencing.
        func::FuncOp callee =
            SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, op.getCalleeAttr());
        if (!callee)
            return signalPassFailure();

        // Now let's create a new function to place the differentiation code into, so it doesn't
        // pollute the current scope. We'll insert the new function after the callee.
        {
            // Insertion guards are useful to store the current IR position (insertion point) on stack,
            // returning to it upon exiting the C++ scope.
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointAfter(callee);

            // Specify the properties of the new function like name, type signature, and visibility.
            std::string fnName = op.getCallee().str() + ".finitediff";
            StringAttr visibility = rewriter.getStringAttr("private");
            // The function type should be identical to the type signature of the grad operation.
            FunctionType fnType = rewriter.getFunctionType(op.getOperandTypes(), op.getResultTypes());

            gradFn = rewriter.create<func::FuncOp>(op.getLoc(), fnName, fnType, visibility, nullptr, nullptr);

            // Now we just to populate the actual body of the function. First create an empty body.
            Block *fnBody = gradFn.addEntryBlock();
            // Move the insertion point to inside the function body.
            rewriter.setInsertionPointToStart(fnBody);
            // Populate the function body.
            populateFiniteDiffMethod(rewriter, op, gradFn);
        }
    }

Symbols are string references to IR objects, which rather than containing a physical reference or
pointer to the actual object, only refer to it by name. In order to dereference a symbol we always
have to look it up in a symbol table. This means that symbols are a bit more flexible and don't
have the same constraints as the `SSA <https://en.wikipedia.org/wiki/Static_single-assignment_form>`_
values used everywhere else in the IR.

To help visualize the process, after this step we would have gone from the IR shown above:

.. code-block:: mlir

    func.func @my_func(%x: f64, %y: f64, %z: f64) -> f64 {
        ...
    }

    %deriv:3 = gradient.grad "fd" @my_func(%x, %y, %z) : (f64, f64, f64) -> (f64, f64, f64)

to the following IR:

.. code-block:: mlir

    func.func @my_func(%x: f64, %y: f64, %z: f64) -> f64 {
        ...
    }

    func.func @my_func.finitediff(f64, f64, f64) -> (f64, f64, f64) {
        <contents of populateFiniteDiffMethod>
    }

    %deriv:3 = gradient.grad "fd" @my_func(%x, %y, %z) : (f64, f64, f64) -> (f64, f64, f64)

Let's fill out the rest of the method. The finite-difference method works by invoking the callee
twice with slightly different parameter values, approximating the partial derivative as follows:

.. math::

    \frac{d}{dx} f(x, y, z) \approx \frac{f(x + h, y, z) - f(x, y, z)}{h}

In code:

.. code-block:: cpp

    void populateFiniteDiffMethod(PatternRewriter &rewriter, GradOp op, func::FuncOp gradFn)
    {
        Location loc = op.getLoc();
        ValueRange callArgs = gradFn.getArguments();

        // We can reuse the same f(x, y, z) evaluation for all partial derivatives.
        func::CallOp callOp = rewriter.create<func::CallOp>(loc, callee, callArgs);

        // Loop through x, y, z to collect the partial derivatives.
        std::vector<Value> gradient;
        for (auto [idx, arg] : llvm::enumerate(callArgs)) {

            FloatAttr hAttr = rewriter.getF64FloatAttr(0.1); // or another small fd parameter
            Value hValue = rewriter.create<arith::ConstantOp>(loc, hAttr);

            Value argPlusH = rewriter.create<arith::AddFOp>(loc, arg, hValue);

            // Make a copy of arguments to replace the argument with it's shifted value.
            std::vector<Value> callArgsForward(callArgs.begin(), callArgs.end());
            callArgsForward[idx] = argPlusH;
            func::CallOp callOpForward =
                rewriter.create<func::CallOp>(loc, callee, callArgsForward);

            // Compute the finite difference.
            Value difference = rewriter.create<arith::SubFOp>(loc, callOpForward.getResult(0), callOp.getResult(0));
            Value partialDerivative = rewriter.create<arith::DivFOp>(loc, difference, hValue);
            gradient.push_back(partialDerivative);
        }

        rewriter.create<func::ReturnOp>(loc, gradient);
    }

Alright, our function should now look something like this:

.. code-block:: mlir

    func.func @my_func.finitediff(%x: f64, %y: f64, %z: f64) -> (f64, f64, f64) {
        %h = arith.constant 0.1 : f64

        %fres = func.call @my_func(%x, %y, %z) : (f64, f64, f64) -> f64

        %xph = arith.addf %x, %h : f64
        %fxph = func.call @my_func(%xph, %y, %z) : (f64, f64, f64) -> f64
        %diffx = arith.subf %fxph, %fres : f64
        %dx = arith.divf %diffx, %h

        %yph = arith.addf %y, %h : f64
        %fyph = func.call @my_func(%x, %yph, %z) : (f64, f64, f64) -> f64
        %diffy = arith.subf %fyph, %fres : f64
        %dy = arith.divf %diffy, %h

        %zph = arith.addf %z, %h : f64
        %fzph = func.call @my_func(%x, %y, %zph) : (f64, f64, f64) -> f64
        %diffz = arith.subf %fzph, %fres : f64
        %dz = arith.divf %diffz, %h

        func.return %dx, %dy, %dz : f64, f64, f64
    }

Finally, we have to amend our rewrite function to invoke the new function we created and delete the
``GradOp`` from the IR:

.. code-block:: cpp

    void FiniteDiffLowering::rewrite(GradOp op, PatternRewriter &rewriter)
    {
        ...
            populateFiniteDiffMethod(rewriter, op, gradFn);
        }

        rewriter.replaceOpWithNewOp<func::CallOp>(op, gradFn, op.getArgOperands());
    }

Note how we can create a new operation, take its results, and use those to replace another operation
in one go. This turns the previous IR:

.. code-block:: mlir

    func.func @my_func(%x: f64, %y: f64, %z: f64) -> f64 {
        ...
    }

    func.func @my_func.finitediff(f64, f64, f64) -> (f64, f64, f64) {
        <contents of populateFiniteDiffMethod>
    }

    %deriv:3 = gradient.grad "fd" @my_func(%x, %y, %z) : (f64, f64, f64) -> (f64, f64, f64)

into:

.. code-block:: mlir

    func.func @my_func(%x: f64, %y: f64, %z: f64) -> f64 {
        ...
    }

    func.func @my_func.finitediff(f64, f64, f64) -> (f64, f64, f64) {
        <contents of populateFiniteDiffMethod>
    }

    %deriv:3 = func.call @my_func.finitediff(%x, %y, %z) : (f64, f64, f64) -> (f64, f64, f64)

.. _catalyst-s-transformation-library:

Catalyst's Transformation Library
=================================

Why don't you try writing a pass of your own? Or have a look at our existing transformations from

- the `quantum dialect <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/lib/Quantum/Transforms>`_,
- the `gradient dialect <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/lib/Gradient/Transforms>`_,
- or the `catalyst utility dialect <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/lib/Catalyst/Transforms>`_.

The pass declarations and headers for transformations are located in the include directory of each
dialect: `quantum <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/include/Quantum/Transforms>`_,
`gradient <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/include/Gradient/Transforms>`_,
and `catalyst <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/include/Catalyst/Transforms>`_.
