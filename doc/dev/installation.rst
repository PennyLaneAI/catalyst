Installation
============


Catalyst is officially supported on Linux (x86_64, aarch64) and macOS (arm64, x86_64) 
platforms, and pre-built binaries are being distributed via the Python Package Index (PyPI) for 
Python versions 3.9 and higher. To install it, simply run the following ``pip`` command:

.. code-block:: console

    pip install pennylane-catalyst

.. warning::

    macOS does not ship with a system compiler by default, which Catalyst depends on. Please
    ensure that `XCode <https://developer.apple.com/xcode/resources/>`_ or the
    ``XCode Command Line Tools`` are installed on your system before using Catalyst.

    The easiest method of installation is to run ``xcode-select --install`` from the Terminal
    app.

Pre-built packages for Windows are not yet available, and compatibility with other platforms is
untested and cannot be guaranteed. If you are using one of these platforms, please
try out our Docker and Dev Container images described in the `next section <#dev-containers>`_.

If you wish to contribute to Catalyst or develop against our runtime or compiler, instructions for
building from source are also included `further down <#minimal-building-from-source-guide>`_.

Dev Containers
--------------


Try out Catalyst in self-contained, ready-to-go environments called
`Dev Containers <https://code.visualstudio.com/docs/devcontainers/containers>`__:

.. image:: https://img.shields.io/static/v1?label=Dev%20Container&message=Launch&color=blue&logo=visualstudiocode&style=flat-square
  :alt: Try Catalyst in Dev Container
  :target: https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/PennyLaneAI/catalyst
  :align: center

| You will need an existing installation of `Docker <https://www.docker.com/>`_,
  `VS Code <https://code.visualstudio.com/>`_, and the VS Code
  `Dev Containers <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>`__
  extension.

If desired, the Docker images can also be used in a standalone fashion:

| `Docker: User Installation <https://github.com/PennyLaneAI/catalyst/blob/main/.devcontainer/Dockerfile>`_
| `Docker: Developer Installation <https://github.com/PennyLaneAI/catalyst/blob/main/.devcontainer/dev/Dockerfile>`_

The user image provides an officially supported environment and automatically installs the latest
release of Catalyst. The developer image only provides the right environment to build Catalyst from
source, and requires launching the post-install script at ``.devcontainer/dev/post-install.sh``
from within the root of the running container.

.. note::

  Due to `a bug <https://github.com/microsoft/vscode-remote-release/issues/8412>`_ in the Dev
  Containers extension, clicking on the "Launch" badge will not prompt for a choice between the User
  and Dev containers. Instead, the User container is automatically chosen.

  As a workaround, you can clone the `Catalyst repository <https://github.com/PennyLaneAI/catalyst>`_
  first, open it as a VS Code Workspace, and then reopen the Workspace in a Dev Container via the
  ``Reopen in Container`` command.



Minimal Building From Source Guide
----------------------------------


Most developers might want to build Catalyst from source instead of using a pre-shipped package. In this section we present a minimal building-from-source installation guide. 

The next section provides a more detailed guide, which we **strongly** recommend the user to read through. Importantly, each component of Catalyst, namely the Python frontend, the MLIR compiler, and the runtime library, can be built and tested indenpendently, which this minimal installation guide does not go over. 


The essential steps are:


.. tabs::

   .. group-tab:: Linux Debian/Ubuntu

      .. warning::
        | If using Anaconda or Miniconda, please make sure to upgrade ``libstdcxx-ng`` via:
        | ``conda install -c conda-forge libstdcxx-ng``
        | If not, you may receive ``'GLIBCXX_3.4.x' not found`` error when running ``make test``.


      .. code-block:: console

        # Install common requirements
        sudo apt install clang lld ccache libomp-dev ninja-build make cmake 

        # Clone the Catalyst repository  
        git clone --recurse-submodules --shallow-submodules https://github.com/PennyLaneAI/catalyst.git

        # Install specific requirements for Catalyst
        cd catalyst
        pip install -r requirements.txt

        # Build Catalyst
        make all

        # Test that everything is built properly
        make test

   .. group-tab:: macOS

      .. code-block:: console

        # Install XCode Command Line Tools and common requirements
        xcode-select --install
        pip install cmake ninja
        brew install libomp

        # Clone the Catalyst repository  
        git clone --recurse-submodules --shallow-submodules https://github.com/PennyLaneAI/catalyst.git

        # Install specific requirements for Catalyst
        cd catalyst
        pip install -r requirements.txt 

        # Build Catalyst
        make all

        # Test that everything is built properly
        make test

These steps should give you the full functionality of Catalyst. 


Detailed Building From Source Guide
-----------------------------------


.. note::
  This section is a detailed building-from-source guide. Some commands in this section has already been included in the minimal guide. 


To build Catalyst from source, developers should follow the instructions provided below for building
all three modules: the Python frontend, the MLIR compiler, and the runtime library.


Requirements
^^^^^^^^^^^^


In order to build Catalyst from source, developers need to ensure the following pre-requisites are
installed and available on the path (depending on the platform):

- The `clang <https://clang.llvm.org/>`_ compiler, `LLD <https://lld.llvm.org/>`_ linker
  (Linux only), `CCache <https://ccache.dev/>`_ compiler cache (optional, recommended), and
  `OpenMP <https://www.openmp.org/>`_.

- The `Ninja <https://ninja-build.org/>`_, `Make <https://www.gnu.org/software/make/>`_, and
  `CMake <https://cmake.org/download/>`_ (v3.20 or greater) build tools.

- `Python <https://www.python.org/>`_ 3.9 or higher for the Python frontend.

- The Python package manager ``pip`` must be version 22.3 or higher.

They can be installed via:


.. tabs::

   .. group-tab:: Linux Debian/Ubuntu

      .. code-block:: console

        sudo apt install clang lld ccache libomp-dev ninja-build make cmake

      .. note::

        If the CMake version available in your system is too old, you can also install up-to-date
        versions of it via ``pip install cmake``.

      .. tabs::

      .. warning::

        If using Anaconda or Miniconda, please make sure to upgrade ``libstdcxx-ng``:

        .. code-block:: console

          conda install -c conda-forge libstdcxx-ng

        If not, you may receive the following error when running ``make test`` because the conda
        environment is using old versions of ``libstdcxx-ng``.

        .. code-block:: console

          'GLIBCXX_3.4.x' not found

   .. group-tab:: macOS

      On **macOS**, it is strongly recommended to install the official XCode Command Line Tools (for ``clang`` & ``make``). The remaining packages can then be installed via ``pip`` and ``brew``:

      .. code-block:: console

        xcode-select --install
        pip install cmake ninja
        brew install libomp



Once the pre-requisites are installed, start by cloning the project repository including all its
submodules:

.. code-block:: console

  git clone --recurse-submodules --shallow-submodules https://github.com/PennyLaneAI/catalyst.git

For an existing copy of the repository without its submodules, they can also be fetched via:

.. code-block:: console

  git submodule update --init --depth=1


All additional build and developer dependencies are managed via the repository's
``requirements.txt`` and can be installed as follows once the repository is cloned:

.. code-block:: console

  pip install -r requirements.txt


.. note::

  Please ensure that your local site-packages for Python are available on the ``PATH`` - watch out
  for the corresponding warning that ``pip`` may give you during installation.

Catalyst
^^^^^^^^

The build process for Catalyst is managed via a series of Makefiles for each component. To build
the entire project from start to finish simply run the following make target from the top level
directory:

.. code-block:: console

  make all

To build each component one by one starting from the runtime, or to build additional backend devices
beyond ``lightning.qubit``, please follow the instructions below.

Runtime
"""""""

By default, the runtime builds and installs all supported backend devices, enabling the execution of
quantum circuits on local simulators and remote services, such as Amazon Braket.
The `PennyLane-Lightning <https://github.com/PennyLaneAI/pennylane-lightning>`__ suite devices require
C++20 standard library features. Older C++ compilers may not support this, so it is recommended to use a
modern compiler with these features.

The full list of supported backends, and additional configuration options, are available in the
`Catalyst Runtime <https://docs.pennylane.ai/projects/catalyst/en/latest/modules/runtime.html>`_
page.

From the root project directory, the runtime can then be built as follows:

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

To make Enzyme libraries discoverable to the compiler:

.. code-block:: console

  export ENZYME_LIB_DIR="$PWD/mlir/Enzyme/build/Enzyme"

To make required tools in ``llvm-project/build``, ``mlir-hlo/mhlo-build``, and
``mlir/build`` discoverable to the compiler:

.. code-block:: console

  export PATH="$PWD/mlir/llvm-project/build/bin:$PWD/mlir/mlir-hlo/mhlo-build/bin:$PWD/mlir/build/bin:$PATH"

Tests
^^^^^

The following target runs all available test suites with the default execution device in Catalyst:

.. code-block:: console

  make test

You can also test each module separately by using running the ``test-frontend``,
``test-dialects``, and ``test-runtime`` targets instead. Jupyter Notebook demos are also testable
via ``test-demos``.

Additional Device Backends
""""""""""""""""""""""""""

The **runtime tests** can be run on additional devices via the same flags that were used to build
them, but using the ``test-runtime`` target instead:

.. code-block:: console

  make test-runtime ENABLE_OPENQASM=ON

.. note::

  The ``test-runtime`` targets rebuilds the runtime with the specified flags. Therefore,
  running ``make runtime OPENQASM=ON`` and ``make test-runtime`` in succession will leave you
  without the OpenQASM device installed.
  In case of errors it can also help to delete the build directory.

The **Python test suite** is also set up to run with different device backends. Assuming the
respective device is available & compatible, they can be tested individually by specifying the
PennyLane plugin device name in the test command:

.. code-block:: console

  make pytest TEST_BACKEND="lightning.kokkos"

AWS Braket devices have their own set of tests, which can be run either locally (``LOCAL``) or on
the AWS Braket service (``REMOTE``) as follows:

.. code-block:: console

  make pytest TEST_BRAKET=LOCAL

Documentation
^^^^^^^^^^^^^

To build and test documentation for Catalyst, you will need to install
`sphinx <https://www.sphinx-doc.org>`_ and other packages listed in ``doc/requirements.txt``:

.. code-block:: console

  pip install -r doc/requirements.txt

Additionally, `doxygen <https://www.doxygen.nl>`_ is required to build C++ documentation, and
`pandoc <https://pandoc.org>`_ to render Jupyter Notebooks.

They can be installed via 


.. tabs::

   .. group-tab:: Linux Debian/Ubuntu

      .. code-block:: console

        sudo apt install doxygen pandoc


   .. group-tab:: macOS

      On **macOS**, `homebrew <https://brew.sh>`_ is the easiest way to install these packages:

      .. code-block:: console

        brew install doxygen pandoc

To generate html files for the documentation for Catalyst:

.. code-block:: console

  make docs

The generated files are located in ``doc/_build/html``

Install a Frontend-Only Development Environment from TestPyPI Wheels
--------------------------------------------------------------------

It is possible to work on the source code repository and test the changes without 
having to compile Catalyst. This is ideal for situations where the changes do not target the 
runtime or the MLIR infrastructure, and only concern to the frontend. It basically 
makes use of the shared libraries already shipped with the TestPyPI Catalyst wheels.

Essential Steps
^^^^^^^^^^^^^^^

.. tabs::

   .. group-tab:: Full CPL Suite

      To activate the development environment, open a terminal and issue the following commands:

      .. code-block:: console

        # Clone the Catalyst repository  
        git clone --recurse-submodules --shallow-submodules https://github.com/PennyLaneAI/catalyst.git

        # Activate the development environment based on the latest TestPyPI wheels.
        # First argument is the name of the Python environment
        # Second argument is the type of Wheel installation. 
        # 'full' installs the whole CPL suite
        cd catalyst
        . ./activate_dev_from_wheel.sh myenv full

   .. group-tab:: Catalyst-Only

    Sometimes the developer has a custom installation of Pennylane or Lightning and would prefer to use
    those ones instead of the ones provided by the TestPyPI Wheels. In that case, to activate the
    development environment, open a terminal and issue the following commands:

      .. code-block:: console

        # Clone the Catalyst repository  
        git clone --recurse-submodules --shallow-submodules https://github.com/PennyLaneAI/catalyst.git

        # Activate the development environment based on the latest TestPyPI wheels.
        # First argument is the name of the Python environment
        # Second argument is the type of Wheel installation. 
        # 'catalyst-only' only installs the Catalyst wheels
        cd catalyst
        . ./activate_dev_from_wheel.sh myenv catalyst-only

To exit the Python virtual environment, type:

.. code-block:: console

  deactivate

How Does it Work?
^^^^^^^^^^^^^^^^^

The provided script first creates and activates a Python virtual environment, so the system Python
configurations do not get affected, nor other virtual environments.

In a second step, it obtains the latest Catalyst wheel from the TestPyPI server and creates hard 
links from the wheel code to the frontend code of the repository, in order to allow working
directly with the frontend repository codebase and at the same time test the changes while
using the installed Catalyst wheel libraries, hence avoiding compilation.

Further Steps
^^^^^^^^^^^^^

If everything goes well, ``git status`` should not report any changed files. 

Before making changes to the frontend, make sure you create a new branch:

.. code-block:: console

  git checkout -b new-branch-name

Once in the new branch, make the wanted changes. Use the IDE of your preference.

You can test the changes by executing your sample code under the same virtual environment you used
with the scripts. As you are actually directly changing the code stored at the Python ``site-packages``
folder, you will be automatically using the shared libraries provided by the Python wheels. Again,
there is no need to compile Catalyst from source.

Commit your changes as usual. Once ready, push the new branch to the remote
repository:

.. code-block:: console
  
  git push -u origin new-branch-name

Now you can go to GitHub and issue a Pull Request based on the new branch.
