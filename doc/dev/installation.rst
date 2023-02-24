Installation
============


Catalyst has been tested on Linux operating systems and the Python frontend is
available as an easy-to-install Python binary via ``pip``. The only
requirements to use Catalyst via Python is `PennyLane
<https://pennylane.ai>`__ and `Python <https://www.python.org/>`_  3.8 or
higher.

Catalyst can then be installed via a single command:

.. code-block:: console

    pip install pennylane-catalyst


We do not currently have binaries available for Windows or MacOS. If you are
using one of these operating systems, or wish to contribute to Catalyst or
develop against our runtime or compiler, please see the instructions below for
building from
source.

Building from source
--------------------


To build Catalyst from its source code, developers should follow the
instructions provided below for building all three modules: the Python
frontend, the MLIR compiler, and runtime library.

Requirements
^^^^^^^^^^^^


In order to build Catalyst from source, developers need to ensure the
following pre-requisites are installed and available on the path:

- The `clang <https://clang.llvm.org/>`_ compiler, `LLD
  <https://lld.llvm.org/>`_ linker, `CCache <https://ccache.dev/>`_ compiler
  cache, and `OpenMP <https://www.openmp.org/>`_.

- The `Ninja <https://ninja-build.org/>`_, `Make
  <https://www.gnu.org/software/make/>`_, and `CMake
  <https://cmake.org/download/>`_ build tools.

- `Python <https://www.python.org/>`_ 3.8 or higher for the Python frontend.

They can be installed on Debian:

.. code-block:: console

  sudo apt install clang lld ccache libomp-dev ninja-build make cmake

The runtime leverages the ``stdlib`` Rust package from the `qir-runner
<https://www.qir-alliance.org/qir-runner>`_ project for the QIR standard
runtime instructions. To build this package from source, the `Rust
<https://www.rust-lang.org/tools/install>`_ toolchain installed via ``rustup``
is required. After installing ``rustup``, the ``llvm-tools-preview`` component
needs to be installed:

.. code-block:: console

  rustup component add llvm-tools-preview

Additionally, the following Python packages for use with Catalyst's build
scripts should be installed:

.. code-block:: console

  pip install -r requirements.txt

Once the pre-requisites are installed, start by cloning the project repository
including all its submodules:

.. code-block:: console

  git clone --recurse-submodules -j8 https://github.com/PennyLaneAI/catalyst.git

For an existing copy of the repository without its submodules, they can also
be fetched via:

.. code-block:: console

  git submodule update --init mlir/llvm-project
  git submodule update --init mlir/mlir-hlo

Catalyst
^^^^^^^^

The build process for Catalyst is managed via a series of Makefiles for each
component. To build the entire project from start to finish simply run the
following make target from the top level directory:

.. code-block:: console

  make all

To build each component one by one starting from the runtime, you can follow
the instructions below.

Runtime
"""""""

By default, the runtime is backed by `PennyLane-Lightning
<https://github.com/PennyLaneAI/pennylane-lightning>`_ and leverages the `QIR
standard library <https://github.com/qir-alliance/qir-runner>`_. Assuming
``libomp-dev`` and the ``llvm-tools-preview`` Rustup component are available,
you can build ``qir-stdlib`` and the runtime from the top level directory:

.. code-block:: console

  make runtime


MLIR Dialects
"""""""""""""

To build the Catalyst MLIR component, along with the necessary `core MLIR
<https://mlir.llvm.org/>`_ and `MLIR-HLO
<https://github.com/tensorflow/mlir-hlo>`_ dependencies, run:

.. code-block:: console

  make mlir

You can also choose to build the custom Catalyst dialects only, with:

.. code-block:: console

  make dialects

Frontend
""""""""

To install the ``pennylane-catalyst`` Python package (the compiler frontend) in editable mode:

.. code-block:: console

  make frontend

Variables
^^^^^^^^^

After following the instructions above, no configuration of environment
variables should be required. However, if you are building Catalyst components
in custom locations, you may need to set and update a few variables on your
system by adjusting the paths in the commands below accordingly.

To make the MLIR bindings from the Catalyst dialects discoverable to the compiler:

.. code-block:: console

  export PYTHONPATH="$PWD/mlir/build/python_packages/quantum:$PYTHONPATH"

To make runtime libraries discoverable to the compiler:

.. code-block:: console

  export RUNTIME_LIB_DIR="$PWD/runtime/build/lib"

To make MLIR libraries discoverable to the compiler:

.. code-block:: console

  export MLIR_LIB_DIR="$PWD/mlir/llvm-project/build/lib"

To make required tools in ``llvm-project/build``, ``mlir-hlo/build``, and
``mlir/build`` discoverable to the compiler:

.. code-block:: console

  export PATH="$PWD/mlir/llvm-project/build/bin:$PWD/mlir/mlir-hlo/build/bin:$PWD/mlir/build/bin:$PATH"

Tests
^^^^^

To check Catalyst modules and the compiler test suites in Catalyst:

.. code-block:: console

  make test

You can also check each module test suite by using ``test-frontend``,
``test-dialects``, and ``test-runtime`` Make targets.
