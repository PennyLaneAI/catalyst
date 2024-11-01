Catalyst Command Line Interface
===============================

Catalyst includes a standalone command-line-interface compiler tool ``catalyst-cli`` that
quantum-compiles MLIR input files into an object file, independent of the Catalyst Python frontend.

This compiler tool combines three stages of compilation:

#. ``quantum-opt``: Performs the MLIR-level optimizations, including quantum optimizations, and
   lowers the input dialect to the LLVM MLIR dialect.
#. ``mlir-translate``: Translates the input LLVM MLIR dialect into LLVM IR.
#. ``llc``: Performs lower-level optimizations on the LLVM IR input and creates the object file.

``catalyst-cli`` runs all three stages under the hood by default, but it also has the ability to run
each stage individually. For example:

.. code-block:: console

    # Creates both the optimized IR and an object file
    catalyst-cli input.mlir -o output.o

    # Only performs MLIR optimizations
    catalyst-cli --tool=opt input.mlir -o llvm-dialect.mlir

    # Only lowers LLVM MLIR dialect input to LLVM IR
    catalyst-cli --tool=translate llvm-dialect.mlir -o llvm-ir.ll

    # Only performs lower-level optimizations and creates object file
    catalyst-cli --tool=llc llvm-ir.ll -o output.o

.. note::

    The Catalyst CLI tool is currently only available when Catalyst is built from source, and is not
    included when installing Catalyst via pip or from wheels.

    After building Catalyst, the ``catalyst-cli`` executable will be available in the
    ``mlir/build/bin/`` directory.

Usage
-----

.. code-block:: console

    catalyst-cli [options] <input file>

Calling ``catalyst-cli`` without any options runs the three compilation stages (``quantum-opt``,
``mlir-translate`` and ``llc``) using all default configurations, and outputs by default an object
file named ``catalyst_module.o``. The name of the output file can be set directly using the ``-o``
option, or by changing the output module name using the ``--module-name`` option (the default module
name is ``catalyst_module``).

Command line options
^^^^^^^^^^^^^^^^^^^^

The complete list of options for the Catalyst CLI tool can be displayed by running ``catalyst-cli --help``.
As this list contains *all* available options, including those for configuring LLVM, the options
most relevant to the usage of the Catalyst CLI tool are covered in more detail below.

``--help``
""""""""""

Show available command-line options and exit.

``--verbose``
"""""""""""""

Emit verbose messages.

``-o <filename>``
"""""""""""""""""

Output filename. If no output filename is provided, and if the ``llc`` compilation step is not run
to produce an object file, the resulting IR is output to stdout.

``--tool=<opt|translate|llc|all>``
""""""""""""""""""""""""""""""""""

Select the tool to run individually. The default is ``all``.

* ``opt``: Run ``quantum-opt`` on the MLIR input.
* ``translate``: Run ``mlir-translate`` on the LLVM MLIR dialect input.
* ``llc``: Run ``llc`` on the LLVM IR input.
* ``all``: Run all of ``opt``, ``translate`` and ``llc`` on the MLIR input.

``--save-ir-after-each=<pass|pipeline>``
""""""""""""""""""""""""""""""""""""""""

Keep intermediate files after each pass or after each pipeline in the compilation. By default, no
intermediate files are saved.

* ``pass``: Keep intermediate files after each transformation/optimization pass.
* ``pipeline``: Keep intermediate files after each pipeline, where a *pipeline* is a sequence of
  transformation/optimization passes.

``--keep-intermediate[=<true|false>]``
""""""""""""""""""""""""""""""""""""""

Keep intermediate files after each pipeline in the compilation. By default, no intermediate files
are saved. Using ``--keep-intermediate`` is equivalent to using ``--save-ir-after-each=pipeline``.

``--catalyst-pipeline=<pipeline1(pass1[;pass2[;...]])[,pipeline2(...)]>``
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Specify the Catalyst compilation pass pipelines.

A pipeline is composed of a semicolon-delimited sequence of one or more transformation or
optimization passes. Multiple pass pipelines can be specified and input as a comma-delimited
sequence of pipelines.

For example, if we wanted to specify two pass pipelines, ``pipe1`` and ``pipe2``, where ``pipe1``
applies the passes ``split-multiple-tapes`` and ``apply-transform-sequence``, and where ``pipe2``
applies the pass ``inline-nested-module``, we would specify this pipeline configuration as:

.. code-block::

    --catalyst-pipeline=pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)

``--workspace=<path>``
""""""""""""""""""""""

The workspace directory where intermediate files are saved. The default is the current working
directory.

Note that the workspace directory must exist before running ``catalyst-cli`` with this option.

``--module-name=<name>``
""""""""""""""""""""""""

The module name used in naming the output file(s). The default is ``"catalyst_module"``. Using the
``-o`` option to specify the output filename overrides this option.

``--async-qnodes[=<true|false>]``
"""""""""""""""""""""""""""""""""

Enable asynchronous QNodes.

``--checkpoint-stage=<stage name>``
"""""""""""""""""""""""""""""""""""

Define a *checkpoint stage*, used to indicate that the compiler should start only after reaching the
given pass.

``--dump-catalyst-pipeline[=<true|false>]``
"""""""""""""""""""""""""""""""""""""""""""

Print (to stderr) the pipeline(s) that will be run.

Examples
^^^^^^^^

To illustrate how to use the Catalyst CLI tool, consider the simple MLIR code, ``my_circuit.mlir``,
which defines a function ``my_circuit`` that implements a single-qubit quantum circuit that applies
the sequence of gates :math:`R_x(\theta) \to H \to H \to R_x(\theta)` to the input qubit for some
rotation angle :math:`\theta`:

.. code-block:: mlir

    module {
      func.func @my_circuit(%in_qubit: !quantum.bit, %angle: f64) -> !quantum.bit {
        %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
        %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
        %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
        %3 = quantum.custom "RX"(%angle) %2 : !quantum.bit
        return %3 : !quantum.bit
      }
    }

We'll use the Catalyst CLI tool to run the ``quantum-opt`` compiler to perform the MLIR-level
optimizations and lower the input to the LLVM MLIR dialect. We'll define a pass pipeline that
applies two quantum-optimization passes:

#. ``remove-chained-self-inverse``, which removes any operations that are applied next to their
   (self-)inverses or adjoint, in this case the two adjacent Hadamard gates.
#. ``merge-rotations``, which combines rotation gates of the same type that act sequentially, in
   this case the two RX gates the become adjacent after the two Hadamard gates have been removed by
   the ``remove-chained-self-inverse`` pass.

To define the pass pipeline, we must supply the name of the function to which each pass applies
using the ``func-name`` argument. To apply these two passes to our ``my_circuit`` function, we can
do so as follows:

.. code-block::

    pipe(remove-chained-self-inverse{func-name=my_circuit};merge-rotations{func-name=my_circuit})

Finally, we'll use the option ``--mlir-print-ir-after-all`` to print the resulting MLIR after each
pass that is applied, and the ``-o`` option to set the name of the output file:

.. code-block::

    catalyst-cli my_circuit.mlir \
        --tool=opt \
        --catalyst-pipeline="pipe(remove-chained-self-inverse{func-name=my_circuit};merge-rotations{func-name=my_circuit})" \
        --mlir-print-ir-after-all \
        -o my_circuit-llvm.mlir

Running this command will output the following intermediate IR to the console:

.. code-block:: mlir

    // -----// IR Dump After RemoveChainedSelfInversePass (remove-chained-self-inverse) //----- //
    module {
      func.func @my_circuit(%arg0: !quantum.bit, %arg1: f64) -> !quantum.bit {
        %out_qubits = quantum.custom "RX"(%arg1) %arg0 : !quantum.bit
        %out_qubits_0 = quantum.custom "RX"(%arg1) %out_qubits : !quantum.bit
        return %out_qubits_0 : !quantum.bit
      }
    }


    // -----// IR Dump After MergeRotationsPass (merge-rotations) //----- //
    module {
      func.func @my_circuit(%arg0: !quantum.bit, %arg1: f64) -> !quantum.bit {
        %0 = arith.addf %arg1, %arg1 : f64
        %out_qubits = quantum.custom "RX"(%0) %arg0 : !quantum.bit
        return %out_qubits : !quantum.bit
      }
    }

and produce a new file ``my_circuit-llvm.mlir`` containing the resulting module in the LLVM MLIR
dialect:

.. code-block:: mlir

    module {
      func.func @my_circuit(%arg0: !quantum.bit, %arg1: f64) -> !quantum.bit {
        %0 = arith.addf %arg1, %arg1 : f64
        %out_qubits = quantum.custom "RX"(%0) %arg0 : !quantum.bit
        return %out_qubits : !quantum.bit
      }
    }

We can see in the intermediate IR after the ``remove-chained-self-inverse`` pass that the two
adjacent Hadamard gates were removed and that the two RX gates were merged into one after the
``merge-rotations`` pass, with the input angle to the single RX gate being the sum of the two input
angles to the original two gates. The result in ``my_circuit-llvm.mlir`` contains the final,
optimized MLIR.

For a list of transformation passes currently available in Catalyst, see the
:ref:`catalyst-s-transformation-library` documentation. The available passes are also listed in the
``catalyst-cli --help`` message.
