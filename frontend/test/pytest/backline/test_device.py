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

"""Unit tests for the ``ftqc.heterogeneous`` device."""

import pennylane as qp
import pytest

from catalyst.api_extensions.target import get_dispatch, get_target
from catalyst.backline import HeterogeneousDevice


def _backline():
    return qp.backline(
        controller=qp.Endpoint(
            "fpga-qpu.ip", role="fpga-qpu", local=False, attrs={"triple": "x86_64-unknown-linux"}
        ),
        coprocessors=(qp.Endpoint("gpu.ip", role="gpu-decoder", local=False),),
        transport="roce",
    )


def _multi_backline():
    """A backline with two same-role (gpu-decoder) coprocessors, individually named."""
    return qp.backline(
        controller=qp.Endpoint("fpga-qpu.ip", role="fpga-qpu"),
        coprocessors=(
            qp.Endpoint("gpu0.ip", role="gpu-decoder", name="gpu-0"),
            qp.Endpoint("gpu1.ip", role="gpu-decoder", name="gpu-1"),
        ),
        transport="roce",
    )


class TestConstruction:
    """The device stores its placement and accepts the stubbed kwargs."""

    def test_stores_backline(self):
        bl = _backline()
        dev = HeterogeneousDevice(wires=49, backline=bl)
        assert dev.backline is bl

    def test_accepts_emulate_and_decoder_kwargs(self):
        # emulate is a recorded no-op in this slice; decoder is stored but not yet dispatched.
        dev = HeterogeneousDevice(wires=2, backline=_backline(), emulate="local", decoder="steane")
        assert dev.backline is not None

    def test_device_kwargs_carries_transport(self):
        # The transport reaches the runtime via device_kwargs -> device_init's rtd_kwargs.
        dev = HeterogeneousDevice(wires=2, backline=_backline())
        assert dev.device_kwargs["transport"] == "roce"


class TestLowering:
    """Construction lowers the backline onto catalyst.target/remote tags."""

    def test_tags_remote_dispatch_from_controller(self):
        dev = HeterogeneousDevice(wires=2, backline=_backline())
        assert get_dispatch(dev).address == "fpga-qpu.ip"

    def test_tags_cross_compile_target_triple(self):
        dev = HeterogeneousDevice(wires=2, backline=_backline())
        tgt = get_target(dev)
        assert tgt.triple == "x86_64-unknown-linux"

    def test_local_controller_is_not_remote_tagged(self):
        bl = qp.backline(
            controller=qp.Endpoint("fpga-qpu.ip", role="fpga-qpu"),  # local by default
            coprocessors=(qp.Endpoint("gpu.ip", role="gpu-decoder"),),
            transport="roce",
        )
        dev = HeterogeneousDevice(wires=2, backline=bl)
        assert get_target(dev) is None
        assert get_dispatch(dev) is None


class TestQueryStubs:
    """Query surface: dummy stubs plus trivial accessors."""

    def test_is_addressable_is_stubbed_true(self):
        assert HeterogeneousDevice(wires=2, backline=_backline()).is_addressable() is True

    def test_resolved_transport(self):
        assert HeterogeneousDevice(wires=2, backline=_backline()).resolved_transport == "roce"


class TestEndpointAddress:
    """Resolve a backline endpoint's address by role (the Step-3 substrate)."""

    def test_resolves_controller_and_coprocessor(self):
        dev = HeterogeneousDevice(wires=2, backline=_backline())
        assert dev.endpoint_address("fpga-qpu") == "fpga-qpu.ip"
        assert dev.endpoint_address("gpu-decoder") == "gpu.ip"

    def test_unknown_role_raises(self):
        dev = HeterogeneousDevice(wires=2, backline=_backline())
        with pytest.raises(ValueError, match="exactly one endpoint"):
            dev.endpoint_address("tpu")


class TestMultiEndpointSelection:
    """Several endpoints can share a role; disambiguate by index (ID) or name."""

    def test_select_by_index(self):
        dev = HeterogeneousDevice(wires=2, backline=_multi_backline())
        assert dev.endpoint_address("gpu-decoder", index=0) == "gpu0.ip"
        assert dev.endpoint_address("gpu-decoder", index=1) == "gpu1.ip"

    def test_select_by_name(self):
        dev = HeterogeneousDevice(wires=2, backline=_multi_backline())
        assert dev.endpoint_address("gpu-decoder", name="gpu-1") == "gpu1.ip"

    def test_ambiguous_role_without_selector_raises(self):
        dev = HeterogeneousDevice(wires=2, backline=_multi_backline())
        with pytest.raises(ValueError, match="found 2"):
            dev.endpoint_address("gpu-decoder")

    def test_index_out_of_range_raises(self):
        dev = HeterogeneousDevice(wires=2, backline=_multi_backline())
        with pytest.raises(ValueError, match="out of range"):
            dev.endpoint_address("gpu-decoder", index=5)

    def test_unknown_name_raises(self):
        dev = HeterogeneousDevice(wires=2, backline=_multi_backline())
        with pytest.raises(ValueError, match="name"):
            dev.endpoint_address("gpu-decoder", name="nope")


class TestEndpointHandle:
    """dev.endpoint(role) yields a dispatch handle usable as kernel.declare(remote=...)."""

    def test_handle_dispatches_to_endpoint_address(self):
        dev = HeterogeneousDevice(wires=2, backline=_backline())
        assert get_dispatch(dev.endpoint("gpu-decoder")).address == "gpu.ip"

    def test_kernel_declare_binds_to_endpoint(self):
        import jax.numpy as jnp
        from jax import ShapeDtypeStruct

        from catalyst import kernel

        dev = HeterogeneousDevice(wires=2, backline=_backline())
        desc = kernel.declare(
            "decode", remote=dev.endpoint("gpu-decoder"), outputs=ShapeDtypeStruct((1,), jnp.int32)
        )
        assert desc.remote_address == "gpu.ip"
        assert desc.remote is True

    def test_remote_endpoint_handle_has_dispatch(self):
        dev = HeterogeneousDevice(wires=2, backline=_backline())  # coprocessor is remote (default)
        handle = dev.endpoint("gpu-decoder")
        assert handle.local is False
        assert get_dispatch(handle).address == "gpu.ip"

    def test_local_endpoint_handle_has_no_remote_dispatch(self):
        bl = qp.backline(
            controller=qp.Endpoint("fpga-qpu.ip", role="fpga-qpu", local=False),
            coprocessors=(qp.Endpoint("gpu.ip", role="gpu-decoder", local=True),),
            transport="roce",
        )
        dev = HeterogeneousDevice(wires=2, backline=bl)
        handle = dev.endpoint("gpu-decoder")
        assert handle.local is True
        assert get_dispatch(handle) is None  # not a remote dispatch target


class TestExecute:
    """Execution is Catalyst-only."""

    def test_execute_raises_not_implemented(self):
        dev = HeterogeneousDevice(wires=2, backline=_backline())
        with pytest.raises(NotImplementedError):
            dev.execute(None, None)


class TestEntryPointResolution:
    """qp.device resolves the device by name via the pennylane.plugins entry point."""

    def test_resolves_by_name(self):
        dev = qp.device("ftqc.heterogeneous", wires=49, backline=_backline(), emulate="local")
        assert isinstance(dev, HeterogeneousDevice)
        assert dev.resolved_transport == "roce"
