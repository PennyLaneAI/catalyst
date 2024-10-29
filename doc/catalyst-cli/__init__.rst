Catalyst CLI
============

Catalyst includes a standalone command-line-interface compiler tool ``catalyst-cli`` that quantum
compiles MLIR input files into an object file, independent of the Python frontend.

This compiler tool combines three stages of compilation:

#. ``quantum-opt``: Performs the MLIR-level optimizations and lowers the input dialect to the LLVM dialect.
#. ``mlir-translate``: Translates the input in the LLVM dialect into LLVM IR.
#. ``llc``: Performs lower-level optimizations and creates the object file.

``catalyst-cli`` runs all three stages under the hood by default, but it also has the ability to run
each stage individually. For example:

.. code-block:: console

    # Creates both the optimized IR and an object file
    catalyst-cli input.mlir -o output.o

    # Only performs MLIR optimizations
    catalyst-cli --tool=opt input.mlir -o llvm-dialect.mlir

    # Only lowers LLVM dialect MLIR input to LLVM IR
    catalyst-cli --tool=translate llvm-dialect.mlir -o llvm-ir.ll

    # Only performs lower-level optimizations and creates object file
    catalyst-cli --tool=llc llvm-ir.ll -o output.o

.. note::

    The Catalyst CLI tool is currently only available when Catalyst is built from source, and is not
    included when installing Catalyst via pip or from wheels.

Usage
-----

Complete usage instructions for the Catalyst CLI tool is available with the ``--help`` option:

.. code-block:: console

    catalyst-cli --help
