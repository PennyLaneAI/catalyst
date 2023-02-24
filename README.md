[![Tests](https://github.com/PennyLaneAI/catalyst/actions/workflows/check-catalyst.yaml/badge.svg)](https://github.com/PennyLaneAI/catalyst/actions/workflows/check-catalyst.yaml)
[![Coverage](https://img.shields.io/codecov/c/github/PennyLaneAI/catalyst/master.svg?logo=codecov&style=flat-square)](https://codecov.io/gh/PennyLaneAI/catalyst)
[![Documentation](https://readthedocs.com/projects/xanaduai-catalyst/badge/?version=latest&token=e6f8607e841564d11d02baef4540523169f95d9c64fcdc656a0ecfd6564203ca)](https://docs.pennylane.ai/projects/catalyst)
[![PyPI](https://img.shields.io/pypi/v/PennyLane-Catalyst.svg?style=flat-square)](https://pypi.org/project/PennyLane-Catalyst)
[![Forum](https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square)](https://discuss.pennylane.ai)
[![License](https://img.shields.io/pypi/l/PennyLane.svg?logo=apache&style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="./doc/_static/catalyst.png#gh-light-mode-only" width="700px">
    <!--
    Use a relative import for the dark mode image. When loading on PyPI, this
    will fail automatically and show nothing.
    -->
  <img src="./doc/_static/catalyst-dark.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt=""/>
</p>

Catalyst is an experimental package that enables just-in-time (JIT) compilation of hybrid
quantum-classical programs.

**Catalyst is currently under heavy development — if you have suggestions on the API or use-cases
you'd like to be covered, please open an GitHub issue or reach out. We'd love to hear about how
you're using the library, collaborate on development, or integrate additional devices and
frontends.**

## Key Features

- Compile the entire quantum-classical workflow, including any optimization loops.

- Use Catalyst alongside PennyLane directly from Python. Simply decorate quantum code and hybrid
  functions with `@qjit`, leading to significant performance improvements over standard Python
  execution.

- Access advanced control flow that supports both quantum and classical instructions.

- Infrastructure for both quantum *and* classical compilation, allowing you to compile quantum
  circuits that contain control flow.

- Built to be end-to-end differentiable.

- Support for the [Lightning](https://github.com/PennyLaneAI/pennylane-lightning/) high
  performance simulator. Additional hardware support, including GPUs and QPUs to come.

## Overview

Catalyst currently consists of the following components:

- [Catalyst Compiler](https://docs.pennylane.ai/projects/catalyst/en/latest/modules/mlir.html).

  The core Catalyst compiler is built using [MLIR](https://mlir.llvm.org/), with the addition of a
  quantum dialect used to represent quantum instructions. This allows for a high-level intermediate
  representation of the classical and quantum components of the program, resulting in advantages
  during optimization. Once optimized, the compiler lowers the representation down to LLVM +
  [QIR](https://www.qir-alliance.org/), and a machine binary is produced.

- [Catalyst
  Runtime](https://docs.pennylane.ai/projects/catalyst/en/latest/modules/runtime.html).

  The
  runtime is a C++ runtime based on QIR that enables the execution of Catalyst-compiled quantum programs.
  Currently, runtime implementations are available for the state-vector simulators
  [Lightning](https://github.com/PennyLaneAI/pennylane-lightning/) and
  [Lightning-Kokkkos](https://github.com/PennyLaneAI/pennylane-lightning-kokkos/). A complete
  list of the quantum instruction set supported by these runtime implementations can be found by
  visiting the Catalyst documentation.

In addition, we also provide a Python frontend for [PennyLane](https://pennylane.ai) and [JAX](https://jax.readthedocs.io):

- [PennyLane JAX
  frontend](https://docs.pennylane.ai/projects/catalyst/en/latest/modules/frontend.html).

  A Python
  library that provides a `@qjit` decorator to just-in-time compile PennyLane hybrid
  quantum-classical programs. In addition, the frontend package provides Python functions for
  defining Catalyst-compatible control flow structures.


## Installation

Catalyst has been tested on Linux operating systems and the Python frontend and C++ binaries are available as an
easy-to-install Python binary via `pip`. The only requirements to use Catalyst via Python is
[PennyLane](https://pennylane.ai) and [Python](https://www.python.org)  3.8 or higher.

Catalyst can then be installed via a single command:

```console
pip install pennylane-catalyst
```

We do not currently have binaries available for Windows or MacOS. If you are using one of these
operating systems, or wish to contribute to Catalyst or develop against our runtime or compiler,
please see the instructions for [building from
source](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/installation.html).

## Trying Catalyst with PennyLane

To get started using Catalyst JIT compiler from Python, check out our [quick start
guide](https://docs.pennylane.ai/projects/catalyst/en/latest/dev/quick_start.html), as well as our
various examples and tutorials in our [documentation](https://docs.pennylane.ai/projects/catalyst).

For an introduction to quantum computing and quantum machine learning, you can also visit the
PennyLane website for [tutorials, videos, and demonstrations](https://pennylane.ai/qml).

## Roadmap

- **Frontend:** As we continue to build out Catalyst, the PennyLane frontend will likely be
  upstreamed into PennyLane proper, providing native JIT functionality built-in to PennyLane. The
  Catalyst compiler and runtime will remain part of the Catalyst project. *If you are interested in
  working on additional frontends for Catalyst, please get in touch.*

- **Compiler:** We will continue to build out the compiler stack, and add quantum compilation
  routines. This includes an API for providing or writing Catalyst-compatible compilation routines.
  In addition, we will be improving the autodifferentiation support, and adding support for
  classical autodiff, additional quantum gradients, and quantum-aware optimization methods.

- **Runtime:** We will be adding support for more devices, including quantum hardware devices. In
  addition, we will be building out support for hetereogeneous execution. *If you are interested in
  working on connecting a quantum device with Catalyst, please get in touch.*

To figure out the details, we need your help — please send us your use cases by starting a
conversation, or trying Catalyst out.

## Contributing to Catalyst

We welcome contributions—simply fork the Catalyst repository, and then make a [pull
request](https://help.github.com/articles/about-pull-requests/) containing your contribution.

We also encourage bug reports, suggestions for new features and enhancements.

## Support

- **Source Code:** https://github.com/PennyLaneAI/catalyst
- **Issue Tracker:** https://github.com/PennyLaneAI/catalyst/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.

We also have a [PennyLane discussion forum](https://discuss.pennylane.ai)—come join the community
and chat with the PennyLane team.

Note that we are committed to providing a friendly, safe, and welcoming environment for all. Please
read and respect the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

PennyLane is the work of [many contributors](https://github.com/PennyLaneAI/catalyst/graphs/contributors).

If you are doing research using Catalyst, please cite our GitHub repo.

## License

Catalyst is **free** and **open source**, released under the Apache License, Version 2.0.
