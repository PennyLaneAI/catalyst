# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains a CudaQDevice and the qjit
entry point.
"""

from importlib.metadata import version
from pathlib import Path

import pennylane as qml


def _check_version_compatibility():
    installed_version = version("cuda_quantum")
    compatible_version = "0.6.0"
    if installed_version != compatible_version:
        msg = f"Compiling with incompatible version cuda_quantum=={installed_version}. "
        msg += f"Please install compatible version cuda_quantum=={compatible_version}."
        raise ModuleNotFoundError(msg)


def cudaqjit(fn=None, **kwargs):
    """A decorator for compiling PennyLane and JAX programs using CUDA Quantum.

    .. important::

        This feature currently only supports CUDA Quantum version 0.6.

    .. note::

        Currently, only the following devices are supported:

        * :class:`softwareq.qpp <SoftwareQQPP>`: a modern C++ statevector simulator
        * :class:`nvidia.statevec <NvidiaCuStateVec>`: The NVIDIA CuStateVec GPU simulator
                                                       (with support for multi-gpu)
        * :class:`nvidia.tensornet <NvidiaCuTensorNet>`: The NVIDIA CuTensorNet GPU simulator
                                                       (with support for matrix product state)

    Args:
        fn (Callable): the quantum or classical function to compile

    Returns:
        QJIT object.

    **Example**

    The compilation is triggered at the call site the
    when the quantum function is executed:

    .. code-block:: python

        dev = qml.device("softwareq.qpp", wires=2)

        @cudaqjit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliY(0))

    >>> circuit(jnp.array([0.5, 1.4]))
    -0.47244976756708373

    From PennyLane, this functionality can also be accessed via

    >>> @qml.qjit(compiler="cuda_quantum")

    Note that CUDA Quantum compilation currently does not have feature parity with Catalyst
    compilation; in particular, AutoGraph, control flow, differentiation, and various measurement
    statistics (such as probabilities and variance) are not yet supported.
    """
    _check_version_compatibility()
    # pylint: disable-next=import-outside-toplevel
    from catalyst.third_party.cuda.catalyst_to_cuda_interpreter import interpret

    if fn is not None:
        return interpret(fn, **kwargs)

    def wrap_fn(fn):
        return interpret(fn, **kwargs)

    return wrap_fn


# Do we need to reimplement apply for every child?
class BaseCudaInstructionSet(qml.devices.QubitDevice):
    """Base instruction set for CUDA-Quantum devices"""

    pennylane_requires = ">=0.34"
    version = "0.1.0"
    author = "Xanadu, Inc."

    # There are similar lines of code in possibly
    # all other list of operations supported by devices.
    # At the time of writing, this warning is raised
    # due to similar lines of code in the QJITDevice
    # pylint: disable=duplicate-code
    operations = [
        "CNOT",
        "CY",
        "CZ",
        "CRX",
        "CRY",
        "CRZ",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "S",
        "T",
        "RX",
        "RY",
        "RZ",
        "SWAP",
        "CSWAP",
    ]
    observables = [
        "PauliX",
        "PauliZ",
    ]
    config = Path(__file__).parent / "cuda_quantum.toml"

    def __init__(self, shots=None, wires=None):
        _check_version_compatibility()
        super().__init__(wires=wires, shots=shots)

    def apply(self, operations, **kwargs):
        """Unused"""
        raise NotImplementedError(
            "This device is only supported with `qml.qjit`."
        )  # pragma: no cover


class SoftwareQQPP(BaseCudaInstructionSet):
    """The SoftwareQ Q++ statevector simulator.

    .. note::

        This device currently only supports QNodes compiled with CUDA Quantum. For a
        high-performance CPU device with support with other compilers, please use
        ``lightning.qubit`` or ``lightning.kokkos``.

    Args:
        shots (None, int): Number of shots to use for measurments and statistics.
            ``None`` corresponds to exact statistics.
        wires (int): Number of wires present on the device.

    **Example**

    .. code-block:: python

        dev = qml.device("softwareq.qpp", wires=2)

        @catalyst.third_party.cuda.cudaqjit
        @qml.qnode(dev)
        def circuit(x):
          qml.RX(x[0], wires=0)
          qml.RY(x[1], wires=1)
          qml.CNOT(wires=[0, 1])
          return qml.expval(qml.PauliY(0))

    >>> circuit(jnp.array([0.5, 1.4]))
    -0.47244976756708373
    """

    short_name = "softwareq.qpp"

    @property
    def name(self):
        """Target name"""
        return "qpp-cpu"


class NvidiaCuStateVec(BaseCudaInstructionSet):
    """The NVIDIA CuStateVec GPU simulator (with support for multi-gpu).

    .. note::

        This device currently only supports QNodes compiled with CUDA Quantum. For a multi-GPU
        device with support with other compilers, please use ``lightning.gpu``.

    Args:
        shots (None, int): Number of shots to use for measurments and statistics.
            ``None`` corresponds to exact statistics.
        wires (int): Number of wires present on the device.
        multi_gpu (bool): Whether to utilize multiple GPUs.

    **Example**

    .. code-block:: python

        dev = qml.device("nvidia.custatevec", wires=2)

        @catalyst.third_party.cuda.cudaqjit
        @qml.qnode(dev)
        def circuit(x):
          qml.RX(x[0], wires=0)
          qml.RY(x[1], wires=1)
          qml.CNOT(wires=[0, 1])
          return qml.expval(qml.PauliY(0))

    >>> circuit(jnp.array([0.5, 1.4]))
    -0.47244976756708373
    """

    short_name = "nvidia.custatevec"

    def __init__(self, shots=None, wires=None, multi_gpu=False):  # pragma: no cover
        self.multi_gpu = multi_gpu
        super().__init__(wires=wires, shots=shots)

    @property
    def name(self):  # pragma: no cover
        """Target name"""
        option = "-mgpu" if self.multi_gpu else ""
        return f"nvidia{option}"


class NvidiaCuTensorNet(BaseCudaInstructionSet):
    """The NVIDIA CuTensorNet GPU simulator (with support for matrix product state)

    .. note::

        This device currently only supports QNodes compiled with CUDA Quantum.

    Args:
        shots (None, int): Number of shots to use for measurments and statistics.
            ``None`` corresponds to exact statistics.
        wires (int): Number of wires present on the device.
        mps (bool): Whether to use matrix product state approximations.

    **Example**

    .. code-block:: python

        dev = qml.device("nvidia.cutensornet", wires=2)

        @catalyst.third_party.cuda.cudaqjit
        @qml.qnode(dev)
        def circuit(x):
          qml.RX(x[0], wires=0)
          qml.RY(x[1], wires=1)
          qml.CNOT(wires=[0, 1])
          return qml.expval(qml.PauliY(0))

    >>> circuit(jnp.array([0.5, 1.4]))
    -0.47244976756708373
    """

    short_name = "nvidia.cutensornet"

    def __init__(self, shots=None, wires=None, mps=False):  # pragma: no cover
        self.mps = mps
        super().__init__(wires=wires, shots=shots)

    @property
    def name(self):  # pragma: no cover
        """Target name"""
        option = "-mps" if self.mps else ""
        return f"tensornet{option}"


__all__ = [
    "cudaqjit",
    "BaseCudaInstructionSet",
    "SoftwareQQPP",
    "NvidiaCuStateVec",
    "NvidiaCuTensorNet",
]
