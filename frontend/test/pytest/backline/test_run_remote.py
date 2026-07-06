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

"""Unit tests for catalyst.run_remote (IR-level checks live in the lit suite)."""

import pennylane as qp

import catalyst
from catalyst.api_extensions.target import get_dispatch, get_target


class TestRunRemote:
    """run_remote tags an ordinary device for remote execution, reusing target/remote."""

    def test_tags_from_address_string(self):
        dev = catalyst.run_remote(qp.device("null.qubit", wires=2), "host:1234")
        assert get_dispatch(dev).address == "host:1234"
        assert get_target(dev) is not None

    def test_returns_the_same_device(self):
        base = qp.device("null.qubit", wires=2)
        assert catalyst.run_remote(base, "host:1234") is base

    def test_tags_from_endpoint(self):
        ep = qp.Endpoint(
            "gpu.ip", role="qpu", local=False, attrs={"triple": "x86_64-unknown-linux"}
        )
        dev = catalyst.run_remote(qp.device("null.qubit", wires=2), ep)
        assert get_dispatch(dev).address == "gpu.ip"
        assert get_target(dev).triple == "x86_64-unknown-linux"
