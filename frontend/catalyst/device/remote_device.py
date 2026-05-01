# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Remote-execution device wrapper.

Wraps pennylane device and tags it with the remote executor's address and
target triple. It cross-compiles the kernel object for that triple and
JIT-links it into the remote `catalyst-executor` over ORC EPC.

Usage:
main.py:
    ```python
    import catalyst
    import pennylane as qp

    dev = catalyst.RemoteDevice(
        "lightning.qubit", wires=2,
        address="192.168.2.9:9997",
        arch="x86_64-linux-gnu"
    )

    @qp.qjit
    @qp.qnode(dev)
    def circuit():
        qp.PauliX(0)
        return qp.expval(qp.PauliZ(0))

    print(circuit())
    ```

shell:
    ```bash
    CATALYST_REMOTE_DEVICE_LIB_LIGHTNING_QUBIT=/path/to/liblightning_qubit_catalyst.so \
        python3 main.py
    ```
"""
import os

import pennylane as qp


def _device_lib_env_key(name: str) -> str:
    """Map a device name to the env var.
    e.g. "lightning.qubit" becomes "CATALYST_REMOTE_DEVICE_LIB_LIGHTNING_QUBIT".
    """
    safe = name.upper().replace(".", "_").replace("-", "_")
    return f"CATALYST_REMOTE_DEVICE_LIB_{safe}"


def _resolve_device_lib(name: str, explicit: str | None) -> str | None:
    """Resolve the executor-side device library path."""
    if explicit:
        return explicit
    return os.environ.get(_device_lib_env_key(name)) or os.environ.get("CATALYST_REMOTE_DEVICE_LIB")


def RemoteDevice(name, *, address, arch, device_lib=None, **kwargs):  # pylint: disable=invalid-name
    """Construct a pennylane device tagged for remote ORC JIT execution."""
    if not isinstance(address, str) or ":" not in address:
        raise ValueError(f"RemoteDevice: address must be 'host:port', got {address!r}")
    if not isinstance(arch, str) or not arch:
        raise ValueError(
            f"RemoteDevice: arch must be a non-empty LLVM target triple, " f"got {arch!r}"
        )

    resolved_device_lib = _resolve_device_lib(name, device_lib)
    if resolved_device_lib is None:
        raise ValueError(
            f"RemoteDevice: cannot determine the remote device library "
            f"set the env var {_device_lib_env_key(name)}"
        )

    dev = qp.device(name, **kwargs)
    dev.catalyst_remote_address = address
    dev.catalyst_remote_arch = arch
    dev.catalyst_remote_device_lib = resolved_device_lib
    return dev
