Installation
============


Catalyst is officially supported on Linux (x86_64) platforms, and pre-built binaries are being
distributed via the Python Package Index (PyPI) for Python versions 3.8 and higher. To install it,
simply run the following ``pip`` command:

.. code-block:: console

    pip install pennylane-catalyst


Pre-built packages for Windows and MacOS are not yet available, and comptability with those
platforms is untested and cannot be guaranteed. If you are using one of these platforms, please
try out our Docker and Dev Container images described in the `next section <#dev-containers>`_.

If you wish to contribute to Catalyst or develop against our runtime or compiler, instructions for
building from source are also included `further down <#building-from-source>`_.

Dev Containers
--------------


Try out Catalyst in self-contained, ready-to-go environments called
`Dev Containers <https://code.visualstudio.com/docs/devcontainers/containers>`_:

.. image:: https://img.shields.io/static/v1?label=Dev%20Container&message=Launch&color=blue&logo=visualstudiocode&style=flat-square
  :alt: Try Catalyst in Dev Container
  :target: https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/PennyLaneAI/catalyst
  :align: center

| You will need an existing installation of `Docker <https://www.docker.com/>`_,
  `VS Code <https://code.visualstudio.com/>`_, and the VS Code
  `Dev Containers <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`_
  extension.

If desired, the Docker images can also be used in a standalone fashion:

| `Docker: User Installation <https://github.com/PennyLaneAI/catalyst/blob/main/.devcontainer/Dockerfile>`_
| `Docker: Developer Installation <https://github.com/PennyLaneAI/catalyst/blob/main/.devcontainer/dev/Dockerfile>`_

The user image provides an officially supported enviroment and automatically installs the latest
release of Catalyst. The developer image only provides the right enviroment to build Catalyst from
source, and requires launching the post-install script at ``.devcontainer/dev/post-install.sh``
from whithin the root of the running container.

.. note::
  Due to `a bug <https://github.com/microsoft/vscode-remote-release/issues/8412>`_ in the Dev
  Containers extension, clicking on the "Launch" badge will not prompt for a choice between the User
  and Dev containers. Instead, the User container is automatically chosen.

  As a workaround, you can clone the `Catalyst repository <https://github.com/PennyLaneAI/catalyst>`_
  first, open it as a VS Code Workspace, and then reopen the Workspace in a Dev Container via the
  ``Reopen in Container`` command.

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
  <https://cmake.org/download/>`_ (v3.20 or greater) build tools.

- `Python <https://www.python.org/>`_ 3.8 or higher for the Python frontend.

- ``pip`` must be version 22.3 or higher.

They can be installed on Debian/Ubuntu via:

.. code-block:: console

  sudo apt install clang lld ccache libomp-dev ninja-build make cmake

.. Note::
  If the CMake version available in your system is too old, you can also install up-to-date
  versions of it via ``pip install cmake``.

The runtime leverages the ``stdlib`` Rust package from the `qir-runner
<https://www.qir-alliance.org/qir-runner>`_ project for standard
QIR runtime instructions. To build this package from source, a `Rust
<https://www.rust-lang.org/tools/install>`_ toolchain installed via ``rustup``
is required. After installing ``rustup``, the ``llvm-tools-preview`` component
needs to be installed:

.. code-block:: console

  rustup component add llvm-tools-preview

All additional build and developer depencies are managed via the repository's ``requirements.txt``
and can be installed as follows:

.. code-block:: console

  pip install -r requirements.txt

Once the pre-requisites are installed, start by cloning the project repository
including all its submodules:

.. code-block:: console

  git clone --recurse-submodules --shallow-submodules -j2 https://github.com/PennyLaneAI/catalyst.git

For an existing copy of the repository without its submodules, they can also
be fetched via:

.. code-block:: console

  git submodule update --init --depth=1

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
<https://github.com/PennyLaneAI/pennylane-lightning>`_
requiring the use of C++20 standard library headers, and leverages the `QIR
standard library <https://github.com/qir-alliance/qir-runner>`_. Assuming
``libomp-dev`` and the ``llvm-tools-preview`` Rustup component are available,
you can build ``qir-stdlib`` and the runtime from the top level directory:

.. code-block:: console

  make runtime

The runtime supports multiple backend devices, enabling the execution of quantum
circuits locally on CPUs and GPUs, and remotely on Amazon Braket NISQ hardware.
A list of supported backends, along with Make arguments for each device, is available in the `Catalyst Runtime <https://docs.pennylane.ai/projects/catalyst/en/latest/modules/runtime.html>`_ page.

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

The following target runs all available test suites in Catalyst:

.. code-block:: console

  make test

You can also test each module separately by using running the ``test-frontend``,
``test-dialects``, and ``test-runtime`` targets instead.
