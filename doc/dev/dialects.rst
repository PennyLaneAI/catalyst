MLIR Dialects in Catalyst
#########################


Introduction to MLIR dialects
=============================

A *dialect* in MLIR is intended to represent a certain abstraction level, computing domain, or
set of thematically-connected program elements, by grouping all necessarily IR objects into
a common namespace. In MLIR, the relevant IR objects include operations, types, attributes, traits,
interfaces, and passes (among others).

A well-scoped dialect can form a reusable component across different compiler IRs, for example the
built-in ``func`` dialect provides everything needed to represent functions (including function
definitions & declarations), call graphs, and inlining transformations.
This means dialects are not monolithic program representations, but instead can be composed with
other dialects to represent a large variety of programs. This structure is especially well-suited to
represent heterogenous programs (like quantum algorithms).

Additionally, dialects are typically organized in an *abstraction hierarchy*, with conversion rules
describing how programs can be lowered from a high-level representation (or abstraction) to a
low-level one. One such conversion rule might involve the decomposition of complex instructions
into a sequence of simpler instructions, or the allocation of memory for various high-level objects.
Alongside of optimization rules, these processes form the core part of program *compilation*.

For additional information on MLIR dialects, and the type of IR objects provided by MLIR, refer to
the following documentation pages:

- `the MLIR language reference <https://mlir.llvm.org/docs/LangRef/>`_: A technical reference for
  the MLIR language (the textual IR format), as well as its concepts and building blocks.
- `built-in MLIR dialects <https://mlir.llvm.org/docs/Dialects/>`_: A list of dialects and
  associated documentation for dialects developed with the MLIR project or contributed upstream.
- `defining MLIR dialects <https://mlir.llvm.org/docs/DefiningDialects/>`_: Documentation describing
  how to define MLIR dialects and all its components.
- `debugging tips <https://mlir.llvm.org/getting_started/Debugging/>`_: Useful reference for any
  kind of MLIR development, especially the debugging flags for `mlir-opt` like tools.

The remainder of this document will guide you through, in concrete terms, how to add a new dialect
to Catalyst.

Catalyst dialect structure
==========================

A strong design principle in the MLIR project is the avoidance of boilerplate code, favoring
a single source of truth to specify IR components. This is achieved via one or more descriptive
languages based on the `TableGen <https://llvm.org/docs/TableGen/index.html>`_ format known from
LLVM. Consequently, dialects, as well as many operations, types, attributes, and other IR objects,
are specified in two parts:

- **a declarative part**: The main way to define new IR objects via a simple language called
  `Operation Definition Specification (ODS) <https://mlir.llvm.org/docs/DefiningDialects/Operations/>`_.
  ODS specifications live in TableGen (``.td``) files, which will automatically generate both C++
  header and implementation files via the build system.

  In Catalyst, TableGen files for dialects can be found in the public include directory, under
  `catalyst/mlir/include/\<DialectName\>/IR <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/include/Quantum/IR>`_.

- **a C++ part**: Additional hand-written logic, and most transformations, are defined directly in C++
  source files. An example might be a storage class for a data type, or the implementation of
  interface methods for an operation declared in ODS.
  Note that while the usage of ODS is strongly recommended, full control over the underlying
  objects can always be achieved by directly working with the relevant C++ classes.

  In Catalyst, hand-written implementations for classes & methods can be found in the source
  directory, under `catalyst/mlir/lib\/<DialectName\>/IR <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/lib/Quantum/IR>`_.

An alternative tutorial on creating a basic MLIR dialect can also be found in the official docs, see
`Creating a Dialect <https://mlir.llvm.org/docs/Tutorials/CreatingADialect/>`_.


Creating a new dialect
======================

Let's see how we can add a new dialect to Catalyst, using OpenQASM as an example! Note that all
auto-generated files and build artifacts mentioned in the guide will only appear at the end of
the `Using the dialect <#using-the-dialect>`_ section when the ``make dialects`` command is run.

Start by creating a new TableGen file, located at ``mlir/include/OpenQASM/OpenQASM.td``, with the
following content:

.. code-block:: tablegen

    include "mlir/IR/DialectBase.td"

    def OpenQASMDialect : Dialect {
        let summary = "An OpenQASM 3 dialect in MLIR.";
        let description = [{
            ...
        }];

        /// This is the namespace of the dialect in MLIR, which is used as a prefix for types and ops.
        let name = "oq";

        /// This is the C++ namespace that the dialect, and all sub-components, get placed in.
        let cppNamespace = "::catalyst::openqasm";

        /// Use the default type printing/parsing hooks, otherwise we would have to explicitly define them.
        let useDefaultTypePrinterParser = 1;
    }

Note the use of ``def`` in TableGen indicates that we are defining a new type of object - under the
hood this results in a new C++ class. To avoid repeating common parts of object definitions, this
object creation mechanism itself supports classes. ``def OpenQASMDialect : Dialect`` defines an
instance of the (built-in) ``Dialect`` TableGen type. In this way, ``OpenQASMDialect`` will inherit
all properties defined for the ``Dialect`` type.

We could also have subtyped the ``Dialect`` type with ``class QuantumDialect : Dialect``, and then
created two kinds of quantum dialects as ``def OpenQASMDialect : QuantumDialect`` and
``def QIRDialect : QuantumDialect`` - if we thought that the two shared some common elements.
Objects defined with ``def`` are terminal and can no longer be subtyped or inherited from.

Similarly, a new data type for our dialect can be added by via the built-in ``TypeDef`` class
(in the same file):

.. code-block:: tablegen

    include "mlir/IR/AttrTypeBase.td"

    class OpenQASM_Type<string name, string nameInIR> : TypeDef<OpenQASMDialect, name, []> {
        let mnemonic = nameInIR;
    }

    def QubitType : OpenQASM_Type<"Qubit", "qubit"> {
        let summary = "A single quantum bit reference.";
    }

TableGen classes accept parameters in angular brackets (``<>``) that can be used in the definition of
class properties, as well as passed on to parent classes.

.. note::

    Do not confuse TableGen classes with C++ classes. Two TableGen objects that inherit
    from the same TableGen class will not share a common base class in C++!

Lastly, let's also add an operation to our dialect (again in the same file), which will allow us to
run a small example at the end.

.. code-block:: tablegen

    include "mlir/IR/OpBase.td"

    class OpenQASM_Op<string nameInIR> : Op<OpenQASMDialect, nameInIR, []>;

    def RZGate : OpenQASM_Op<"RZ"> {
        let summary = "A single-qubit rotation around the Z-axis by an angle θ.";

        let arguments = (ins
            F64:$theta,
            QubitType:$qubit
        );

        let results = (outs
        );

        let assemblyFormat = [{
            `(` $theta `)` $qubit attr-dict `:` type($qubit)
        }];
    }

Operations are primarily defined via their *arguments* and *results*. In the IR, argument & result
values are what organize operations into a graph (the so-called SSA graph), which encodes the flow
of data through the program. The MLIR guide
`Understanding the IR Structure <https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure>`_
can be helpful in obtaining a deeper understanding of this concept.

Further, operations can define nested regions with additional operations. How the nested region
will be executed is entirely up to the operation and its lowering mechanism. The concept of nesting
operations is used in many places in MLIR. The built-in operations ``module`` and ``func`` are
themselves just implemented as regular operations with a nested region. The structured control flow
dialect (SCF) also uses it to represent branching and looping in a much more intuitive fashion
than LLVM.

Lastly, we also defined a custom syntax (``assemblyFormat``) for our operation. MLIR provides two
ways of representing operations in its textual assembly format:

- **generic assembly format**: This format is a one-to-one mapping from how MLIR objects are
  represented in memory, and contains all necessary information to uniquely represent an MLIR
  program with it. As a consequence, this format can be used to parse and print operations from any
  dialect, even unknown ones!

  The generic assembly format can be very useful for debugging, as it more truthfully represents the
  IR state. It also suffers less from crashing in the case of an invalid IR state.

- **pretty assembly format**: This format can be fully customized (with some restrictions) by the
  dialect designer. Generally speaking the IR can be much more human-readable when printed in this
  form. Common improvements include imitating a particular syntax (e.g. indexed array access),
  structuring operands into groups, and omitting redundant type information.

More information on defining operations and other dialect objects can be found in the
`dialects <https://mlir.llvm.org/docs/DefiningDialects>`_,
`attributes & types <https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/>`_,
and `operations <https://mlir.llvm.org/docs/DefiningDialects/Operations/>`_
pages of the MLIR documentation.


Building the dialect
====================

The easiest way to build dialects is to use predefined CMake functions provided by MLIR for this
purpose. The build system will then generate C++ code based on the given TableGen definitions.

Add a new file ``mlir/include/OpenQASM/CMakeLists.txt`` with the following content:

.. code-block:: cmake

    add_mlir_dialect(OpenQASM oq)

The first argument, ``OpenQASM``, has to match the name of the main TableGen file of our dialect
exactly ("main" because TableGen files can be included in other TableGen files, and it can be
useful to organize definitions across several files) - in our case that file is ``OpenQASM.td``.
The second argument, ``oq``, has to match the chosen dialect name (or prefix) in MLIR.

With the provided TableGen definitions, CMake will generate a set of C++ files as follows:

- ``OpenQASMDialect.h.inc``: A C++ header file for dialect-related class declarations.
- ``OpenQASMDialect.cpp.inc``: A C++ source file for (certain) dialect method definitions.
- ``OpenQASMTypes.h.inc``: A C++ header file with declarations of our dialect types.
- ``OpenQASMTypes.cpp.inc``: A C++ source file with definitions of type-related methods, including
  for example how to print & parse a given type (thanks to ``useDefaultTypePrinterParser = 1``).
- ``OpenQASM.h.inc``: A C++ header file with declarations for all our dialect operations.
- ``OpenQASM.cpp.inc``: A C++ source file with definitions of operation methods, such as printing
  & parsing as well as instantiating new operations (MLIR calls these operation *builders*).

Depending on the features provided by a dialect, you may see additional files here, such as for
attributes, interfaces, and other types of MLIR objects.

The suffix ``.inc`` indicates that the files have been automatically generated, and are by themselves
not sufficient to produce a library with our dialect. Instead, all these files are meant to be
included in a few header and source files of our own.

Let's start with a public header file for our dialect. Other Catalyst code can then include this
header to manipulate objects from our dialect. Create a file ``mlir/include/OpenQASM/OpenQASM.h``
with the following content:

.. code-block:: c++

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

Here we directly included declarations for all the object types we defined in a single header.
Note that some auto-generated files allow you to selectively include code via pre-processor flags,
as done here for types and operations. It can be a good idea to directly look into ``.inc`` to
understand the type of code they provide.

Lastly, let's create a main source file for our dialect at ``mlir/lib/OpenQASM/OpenQASM.cpp``:

.. code-block:: c++

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

Besides ensuring that the right MLIR headers for our code are included, we mainly need to insert
all the auto-generated C++ source files, just like we did for the dialect header. The snippet also
demonstrates that some methods need to be manually implemented, like the dialect initialization
function ``OpenQASMDialect::initialize()``. Other methods that are typically manually added include
operation verifiers and operation folding & canonicalization methods.

The accompanying CMake script ``mlir/lib/OpenQASM/CMakeLists.txt`` will generate a build target
that other Catalyst components can depend on:

.. code-block:: cmake

    add_mlir_library(MLIROpenQASM
        OpenQASM.cpp

        DEPENDS
        MLIROpenQASMIncGen
    )

Note the naming scheme: ``MLIROpenQASM`` is a name of our choice for the dialect build target, while
``MLIROpenQASMIncGen`` is a target automatically provided by the `add_mlir_dialect` function from the
provided TableGen file name (``OpenQASM``). The latter represents the generation of C++ files from
TableGen.

.. warning::

    For any newly added ``CMakeLists.txt``, be sure to add it to its parent CMake file with
    ``add_subdirectory(<name of new folder>)``. In this case, both ``mlir/include/CMakeLists.txt``
    and ``mlir/lib/CMakeLists.txt`` will need to be updated with ``add_subdirectory(OpenQASM)``.


Using the dialect
=================

MLIR's standard tool for testing dialects and compiler passes is the ``opt`` tool (inherited from
LLVM). The tool parses a program in the textual MLIR format, applies arbitrary passes, and prints
the transformed program back out. Parsing and printing out a program without any transformations
is also referred to as "round-tripping". Let's see if we can pass this first test with our dialect!

Catalyst comes with its own version of the opt tool, ``quantum-opt``, preloaded with all builtin
MLIR dialects and transformations, as well all additional compiler components developed for
Catalyst specifically. Add the following two lines to the ``mlir/tools/quantum-opt/quantum-opt.cpp``
file:

.. code-block:: c++

    // ...
    #include "OpenQASM/OpenQASM.h"  // add me

    int main(int argc, char **argv)
    {
        // ...
        registry.insert<catalyst::openqasm::OpenQASMDialect>();  // add me

        // ...
    }

Similarly, update the corresponding ``mlir/tools/quantum-opt/CMakeLists.txt`` to include the build
target for our dialect as a dependent library:

.. code-block:: cmake

    # ...
    set(LIBS
        # ...
        MLIROpenQASM  # add me
    )

    # ...

That's it! We can now build our additions with the rest of the dialects and test them out. Assuming
Catalyst has already been built successfully at least once, simply run:

.. code-block:: console

    make dialects

Save the following test file somewhere and run it through the ``quantum-opt`` tool:

.. code-block:: mlir

    func.func @my_circuit(%q0 : !oq.qubit) {
        %phi = arith.constant 0.3 : f64

        oq.RZ(%phi) %q0 : !oq.qubit

        func.return
    }

.. code-block:: console

    ./mlir/build/bin/quantum-opt my_test_file.mlir

You should see the same code in the input file printed back out to you:

.. code-block::

    func.func @my_circuit(%q0 : !oq.qubit) {
        %phi = arith.constant 0.3 : f64

        oq.RZ(%phi) %q0 : !oq.qubit

        func.return
    }

.. note::

    If you are encoutering issues, or would like to quickly try out the dialect described in this
    guide, you can have a look at or cherry-pick this commit which includes all changes described
    above: https://github.com/PennyLaneAI/catalyst/commit/e36d435c209a32f06715f3e34ac896a0a35aa92c


Build your own
==============

To take your dialect to the next level, be sure to also check out the
`Catalyst transformation guide <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/transforms.html>`_
for information on how to write transformation passes for Catalyst.

For additional inspiration and reference implementations, don't forget to check out the existing
dialects at `catalyst/mlir/include <https://github.com/PennyLaneAI/catalyst/tree/main/mlir/include>`_.
