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

"""The ``ftqc.heterogeneous`` device.

The device is the thin wrapper over Catalyst's existing remote-execution frontend: it
takes a :class:`~pennylane.backline.Backline` placement and lowers it onto the ``catalyst.target`` /
``catalyst.remote`` tags that the cross-compile and dispatch passes consume. Execution is
Catalyst-only (``@qjit``); the runtime library it targets is produced by a later increment.
"""

import platform
from typing import Optional

from pennylane import CompilePipeline
from pennylane.devices import Device, ExecutionConfig

from catalyst.api_extensions.target import RemoteDispatch, run_remote


class _EndpointHandle:
    """A dispatch target for a single backline endpoint.

    Carries the endpoint's locality, and its address for a remote endpoint, which is handed to ``kernel.declare(remote=...)`` and resolved through the standard ``get_dispatch`` path. A local endpoint carries does not carry a dispatch: its kernel is called
    locally.
    """

    def __init__(self, endpoint):
        self.host = endpoint.host
        self.role = endpoint.role
        self.name = endpoint.name
        self.local = endpoint.local
        if not endpoint.local:
            self._catalyst_dispatch = RemoteDispatch(address=endpoint.host)


class HeterogeneousDevice(Device):
    """A remote-capable device for heterogeneous execution over a backline placement.

    Args:
        wires (int): Number of wires.
        backline (Backline): The placement (controller, coprocessors, transport), constructed via
            :func:`pennylane.backline`.
        emulate (str): Emulate locally. Defaults to `"False"`. Not yet implemented.
        decoder: The decoder to use. Defaults to ``None``.
    """

    def __init__(self, wires, backline, emulate="local", decoder=None, **kwargs):
        self._backline = backline
        self._emulate = emulate
        self._decoder = decoder
        super().__init__(wires=wires, **kwargs)

        # Carry the transport to the runtime via device_init's rtd_kwargs (device_kwargs -> rtd_kwargs).
        self.device_kwargs = {"transport": str(backline.transport)}

        controller = backline.controller
        if not controller.local:
            run_remote(self, controller)

    @property
    def backline(self):
        """The :class:`~pennylane.backline.Backline` placement the device was configured with."""
        return self._backline

    @property
    def resolved_transport(self):
        """The transport in use (the backline's transport)."""
        return self._backline.transport

    def _select_endpoint(self, role, *, index=None, name=None):
        """Resolve one backline endpoint with ``role``, disambiguated by ``name`` or ``index``.

        Raises:
            ValueError: If the selection does not resolve to exactly one endpoint.
        """
        endpoints = (self._backline.controller, *self._backline.coprocessors)
        matches = [ep for ep in endpoints if ep.role == role]
        if name is not None:
            named = [ep for ep in matches if ep.name == name]
            if len(named) != 1:
                raise ValueError(
                    f"expected exactly one endpoint with role {role!r} and name {name!r}, "
                    f"found {len(named)}"
                )
            return named[0]
        if index is not None:
            if not 0 <= index < len(matches):
                raise ValueError(
                    f"endpoint index {index} out of range for role {role!r} "
                    f"({len(matches)} endpoint(s))"
                )
            return matches[index]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one endpoint with role {role!r}, found {len(matches)}; "
                f"pass index= or name= to disambiguate"
            )
        return matches[0]

    def endpoint_address(self, role, *, index=None, name=None):
        """Return the host of the backline endpoint with ``role`` (see :meth:`_select_endpoint`)."""
        return self._select_endpoint(role, index=index, name=name).host

    def endpoint(self, role, *, index=None, name=None):
        """Return a dispatch handle for the backline endpoint with ``role``.

        Several endpoints may share a role; pass ``name`` or ``index`` to pick one. Hand the handle to
        ``kernel.declare(remote=...)`` to dispatch a kernel to a **remote** endpoint (e.g. a GPU
        decoder on another host), resolved via the standard ``get_dispatch`` path::

            dec = kernel.declare("decode", remote=dev.endpoint("gpu-decoder"), outputs=...)

        A **local** endpoint's handle carries no dispatch (``handle.local is True``): its kernel is
        called locally, not remote-dispatched.
        """
        return _EndpointHandle(self._select_endpoint(role, index=index, name=name))

    def is_addressable(self):
        """Whether the current process can reach the endpoint. Stub: always ``True``."""
        return True

    def is_remote(self):
        """Whether the device executes remotely. Stub: always ``True``."""
        return True

    @staticmethod
    def get_c_interface():
        """Return ``(device_name, runtime_library_path)``.

        Not yet implemented.
        """
        system_extension = ".dylib" if platform.system() == "Darwin" else ".so"
        return "ftqc_heterogeneous", f"librtd_backline{system_extension}"

    def preprocess(self, execution_config: Optional[ExecutionConfig] = None):
        """Return the device transform program and the (unchanged) execution config."""
        if execution_config is None:
            execution_config = ExecutionConfig()
        return CompilePipeline(), execution_config

    def execute(self, circuits, execution_config):
        """Execution is Catalyst-only; there is no Python execution path."""
        raise NotImplementedError("The ftqc.heterogeneous device only supports Catalyst (@qjit).")
