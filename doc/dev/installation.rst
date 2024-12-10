Installation
============


Catalyst is officially supported on Linux (x86_64, aarch64) and macOS (arm64, x86_64)
platforms, and pre-built binaries are being distributed via the Python Package Index (PyPI) for
Python versions 3.10 and higher. To install it, simply run the following ``pip`` command:

.. code-block:: console

    pip install pennylane-catalyst

.. warning::

    macOS does not ship with a system compiler by default, which Catalyst depends on. Please
    ensure that `XCode <https://developer.apple.com/xcode/resources/>`_ or the
    ``XCode Command Line Tools`` are installed on your system before using Catalyst.

    The easiest method of installation is to run ``xcode-select --install`` from the Terminal
    app.

Pre-built packages for Windows are not yet available, and compatibility is untested and cannot
be guaranteed. If you would like to use Catalyst on Windows, we recommend trying the
`WSL <https://learn.microsoft.com/windows/wsl/>`_.

If you wish to contribute to Catalyst or develop against our runtime or compiler, instructions for
building from source are detailed `below <#minimal-building-from-source-guide>`_.


Minimal Building From Source Guide
----------------------------------


This is an abbreviated set of instructions that can be copy-pasted into the terminal of most
common systems. For information on pre-requisites, how to build individual components, or if
you are encoutering issues, please consult the detailed guide
`in the next section <#detailed-building-from-source-guide>`_.


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

        # If not present yet, install Homebrew (https://brew.sh/)
        brew install libomp ccache gfortran

        # Add ccache drop-in compiler replacements to the PATH
        export PATH=/usr/local/opt/ccache/libexec:$PATH

        # Clone the Catalyst repository
        git clone --recurse-submodules --shallow-submodules https://github.com/PennyLaneAI/catalyst.git

        # Install specific requirements for Catalyst
        cd catalyst
        pip install -r requirements.txt

        # Build Catalyst
        make all

        # Test that everything is built properly
        make test


Detailed Building From Source Guide
-----------------------------------


To build Catalyst from source, developers should follow the instructions provided below for building
all three modules: the Python frontend, the MLIR compiler, and the runtime library.


Requirements
^^^^^^^^^^^^


In order to build Catalyst from source, developers need to ensure the following pre-requisites are
installed and available on the path (depending on the platform):

- The `clang <https://clang.llvm.org/>`_ compiler, `LLD <https://lld.llvm.org/>`_ linker
  (Linux only), `CCache <https://ccache.dev/>`_ compiler cache (optional, recommended), and
  `OpenMP <https://www.openmp.org/>`_. Additionaly, the
  `GFortran <https://fortran-lang.org/en/learn/os_setup/install_gfortran/>`_ compiler is
  required on ARM macOS systems.

- The `Ninja <https://ninja-build.org/>`_, `Make <https://www.gnu.org/software/make/>`_, and
  `CMake <https://cmake.org/download/>`_ (v3.20 or greater) build tools.

- `Python <https://www.python.org/>`_ 3.10 or higher for the Python frontend.

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

      On **macOS**, it is strongly recommended to install the official XCode Command Line Tools (for ``clang`` & ``make``).
      The remaining packages can then be installed via ``pip`` and ``brew``.
      If ``brew`` is not present yet, install it from https://brew.sh/:

      .. code-block:: console

        xcode-select --install
        pip install cmake ninja
        brew install libomp ccache gfortran
        export PATH=/usr/local/opt/ccache/libexec:$PATH



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

Known Issues
------------

.. tabs::

   .. group-tab:: Linux Debian/Ubuntu

      If you get this error:

      .. code-block:: console

        cannot find -lstdc++: No such file or directory

      you might need to install a recent version of ``libstdc``. E.g.:

      .. code-block:: console

        sudo apt install libstdc++-12-dev

      (See user's report `here <https://discourse.llvm.org/t/usr-bin-clang-is-not-able-to-compile-a-simple-test-program/72889/3>`_)

      .. raw:: html

        <hr>

      Under Ubuntu 24.04, if you get this error:

      .. code-block:: console

        fatal error: 'Python.h' file not found

      you might need to install the Python Dev package:

      .. code-block:: console

        sudo apt install python3-dev

      (See user's report `here <https://github.com/PennyLaneAI/catalyst/issues/1084>`_)

   .. group-tab:: macOS

      If using Anaconda or Miniconda, you might need to set up the PYTHON environment variable
      with the path to the Conda Python binary. E.g.:

      .. code-block:: console

        export PYTHON=/Users/<username>/anaconda3/envs/<envname>/bin/python

      If not, PyTest might try to use the default Python binary: ``/usr/bin/python3``.
      (See user's report `here <https://github.com/PennyLaneAI/catalyst/issues/377>`_)

Install a Frontend-Only Development Environment from TestPyPI Wheels
--------------------------------------------------------------------

It is possible to work on the source code repository and test the changes without
having to compile Catalyst. This is ideal for situations where the changes do not target the
runtime or the MLIR infrastructure, and only concern the frontend. It basically
makes use of the shared libraries already shipped with the TestPyPI Catalyst wheels.

Essential Steps
^^^^^^^^^^^^^^^

To activate the development environment, open a terminal and issue the following commands:

.. code-block:: console

  # Clone the Catalyst repository without submodules, as they are not needed for frontend
  # development
  git clone git@github.com:PennyLaneAI/catalyst.git

  # Setup the development environment based on the latest TestPyPI wheels.
  # Please provide a path for the Python virtual environment
  cd catalyst
  bash ./setup_dev_from_wheel.sh /path/to/virtual/env

  # Activate the Python virtual environment
  source /path/to/virtual/env/bin/activate

To exit the Python virtual environment, type:

.. code-block:: console

  deactivate

Special Considerations
^^^^^^^^^^^^^^^^^^^^^^

Catalyst dev wheels are tied to fixed versions of PennyLane and Lightning, which are installed
together as a bundle. If you want to use different versions of Pennylane or Lightning, reinstall the
desired versions after having run the script:

.. code-block:: console

  python -m pip install pennylane==0.*.*
  python -m pip install pennylane-lightning==0.*.*

If you require the Catalyst repository with all its submodules, clone it this way:

.. code-block:: console

  git clone --recurse-submodules --shallow-submodules git@github.com:PennyLaneAI/catalyst.git

How Does it Work?
^^^^^^^^^^^^^^^^^

The provided script first creates and activates a Python virtual environment, so the system Python
configurations do not get affected, nor other virtual environments.

In a second step, it obtains the latest Catalyst wheel from the TestPyPI server and creates hard
links from the wheel code to the frontend code of the repository, in order to allow working
directly with the frontend code of the repository and at the same time test the changes while
using the installed Catalyst wheel libraries, hence avoiding compilation.

Further Steps
^^^^^^^^^^^^^

If everything goes well, ``git status`` should not report any changed files.

Before making changes to the frontend, make sure you create a new branch:

.. code-block:: console

  git checkout -b new-branch-name

Once in the new branch, make the wanted changes. Use the IDE of your preference.

You can test the changes by executing your sample code under the same virtual environment you used
with the scripts. As files in the repository are hard-linked to the Wheel code, you are actually
changing the code stored at the Python ``site-packages`` folder as well, and you will be automatically
using the shared libraries provided by the Python wheels. Again, there is no need to compile Catalyst
from source.

You can commit your changes as usual. Once ready, push the new branch to the remote
repository:

.. code-block:: console

  git push -u origin new-branch-name

Now you can go to GitHub and issue a Pull Request based on the new branch.
