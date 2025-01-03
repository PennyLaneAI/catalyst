Catalyst Command Line Interface
===============================

Catalyst includes a standalone command-line-interface compiler tool ``catalyst-cli`` that
compiles quantum programs written in our MLIR dialects into an object file,
independent of the Catalyst Python frontend.

This compiler tool combines three stages of compilation:

#. ``quantum-opt``: Performs the MLIR-level optimizations, including quantum optimizations, and
   translates the input dialect to the `LLVM dialect <https://mlir.llvm.org/docs/Dialects/LLVM/>`_.
#. ``mlir-translate``: Translates the input LLVM dialect into
   `LLVM IR <https://llvm.org/docs/LangRef.html>`_.
#. ``llc``: Performs lower-level optimizations on the LLVM IR input and creates the object file.

``catalyst-cli`` runs all three stages under the hood by default, but it also has the ability to run
each stage individually. For example:

.. code-block:: console

    # Creates both the optimized IR and an object file
    catalyst-cli input.mlir -o output.o

    # Only performs MLIR optimizations and translates to LLVM dialect
    catalyst-cli --tool=opt input.mlir -o llvm-dialect.mlir

    # Only lowers LLVM dialect input to LLVM IR
    catalyst-cli --tool=translate llvm-dialect.mlir -o llvm-ir.ll

    # Only performs lower-level optimizations and creates object file (object.o)
    catalyst-cli --tool=llc llvm-ir.ll -o output.ll --module-name object

.. note::

    If catalyst is built from source, the (``catalyst-cli``) executable would be located in 
    the ``mlir/build/bin/`` directory relative to the root of your Catalyst source directory.

    If building Catalyst via pip or from wheels, the executable qould be located 
    at ``catalyst/bin/`` directory relative to the environment’s installation directory.

Usage
-----

.. code-block:: console

    catalyst-cli [options] <input file>

Calling ``catalyst-cli`` without any options runs the three compilation stages (``quantum-opt``,
``mlir-translate`` and ``llc``) using all default configurations, and outputs by default an object
file named ``catalyst_module.o``. The name of the output file can be set by changing the output 
module name using the ``--module-name`` option (the default module name is ``catalyst_module``).

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

Output IR filename. If no output filename is provided, the resulting IR is output to stdout.

``--tool=<opt|translate|llc|all>``
""""""""""""""""""""""""""""""""""

Select the tool to run individually. The default is ``all``.

* ``opt``: Run ``quantum-opt`` on the MLIR input.
* ``translate``: Run ``mlir-translate`` on the LLVM dialect input.
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

``--{passname}``
"""""""""""""""

Enable a specific pass. For example, to enable the ``remove-chained-self-inverse`` pass, use
``--remove-chained-self-inverse``.

Catalyst's main ``mlir`` stage is split up into a sequence of pass pipelines that can also be run
individually via this option. In that case, the name of the pipeline is substituted for the pass
name. Currently, the following pipelines are available:
``enforce-runtime-invariants-pipeline``,
``hlo_lowering-pipeline``,
``quantum-compilation-pipeline``,
``bufferization-pipeline``,
``llvm-dialect-lowring-pipeline``, and finally 
``default-catalyst-pipeline`` which encompasses all the above as the default pipeline used by the
Catalyst CLI tool if no pass option is specified.

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

    --catalyst-pipeline="pipe1(split-multiple-tapes;apply-transform-sequence),pipe2(inline-nested-module)"

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
given stage. The stages that are currently available are:
* MLIR:: ``mlir`` (start with first MLIR stage), ``{pipeline}`` such as any of the built-in pipeline names
  described under the ``--{passname}`` option, OR any custom pipeline names if the
  ``--catalyst-pipeline={pipeline(...),...}`` option is used.
* * LLVM: ``llvm_ir`` (start with first LLVM stage), ``CoroOpt``, ``O2Opt``, ``Enzyme``.
  Note that ``CoroOpt`` (Coroutine lowering), ``O2Opt`` (O2 optimization), and ``Enzyme``
  (automatic differentiation) passes are only run conditionally as needed.

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
optimizations and translate the input to the LLVM dialect. We'll define a pass pipeline that applies
two quantum-optimization passes:

#. ``remove-chained-self-inverse``, which removes any operations that are applied next to their
   (self-)inverses or adjoint, in this case the two adjacent Hadamard gates.
#. ``merge-rotations``, which combines rotation gates of the same type that act sequentially, in
   this case the two RX gates the become adjacent after the two Hadamard gates have been removed by
   the ``remove-chained-self-inverse`` pass.

To apply these two passes to our ``my_circuit`` function, we can do so as follows:

.. code-block::

    pipe(remove-chained-self-inverse;merge-rotations)

Finally, we'll use the option ``--mlir-print-ir-after-all`` to print the resulting MLIR after each
pass that is applied, and the ``-o`` option to set the name of the output IR file:

.. code-block::

    catalyst-cli my_circuit.mlir \
        --tool=opt \
        --catalyst-pipeline="pipe(remove-chained-self-inverse;merge-rotations)" \
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

and produce a new file ``my_circuit-llvm.mlir`` containing the resulting module in the LLVM dialect:

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

MLIR Plugins
------------

``mlir-opt``-like tools are able to take plugins as inputs.
These plugins are shared objects that include dialects and passes written by third parties.
This means that you can write dialects and passes that can be used with ``catalyst-cli`` and ``quantum-opt``.

As an example, the `LLVM repository includes a very simple plugin <https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone/standalone-plugin>`_.
To build it, simply run ``make plugin`` and the standalone plugin
will be built in the root directory of the Catalyst project.

With this, you can now run your own passes by using the following flags:

``catalyst-cli --load-dialect-plugin=$YOUR_PLUGIN --load-pass-plugin=$YOUR_PLUGIN $YOUR_PASS_NAME file.mlir``

Concretely for the example plugin, you can use the following command:

``catalyst-cli --tool=opt --load-pass-plugin=standalone/build/lib/StandalonePlugin.so --load-dialect-plugin=standalone/build/lib/StandalonePlugin.so --pass-pipeline='builtin.module(standalone-switch-bar-foo)' a.mlir``
