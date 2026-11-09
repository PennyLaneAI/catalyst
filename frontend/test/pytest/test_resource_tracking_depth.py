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

"""Tests for the circuit depth tracked by null.qubit for classically controlled operations."""

import json

import pennylane as qp
import pytest

from catalyst import qjit

CLASSICAL_CONTROL_PASS = "instrument-classical-control"


def _bufferization_passes(qjit_fn):
    """Return the passes of the bufferization stage the program was compiled with."""
    return dict(qjit_fn.compile_options.get_pipelines())["BufferizationStage"]


def _track(body, capture_mode, tmp_path, wires=3):
    """Execute ``body`` on null.qubit with depth tracking, returning the tracked resources."""
    filename = tmp_path / "resources.json"
    dev = qp.device(
        "null.qubit",
        wires=wires,
        track_resources=True,
        resources_filename=str(filename),
        compute_depth=True,
    )

    @qjit(capture=capture_mode)
    @qp.qnode(dev)
    def circuit():
        body()
        return qp.probs(wires=0)

    circuit()
    assert CLASSICAL_CONTROL_PASS in _bufferization_passes(circuit)

    with filename.open("r", encoding="utf-8") as f:
        return json.load(f)


class TestPipeline:
    """Test that classical control is only instrumented when the circuit depth is tracked."""

    def test_tracking_depth(self, tmp_path):
        """Test that the pass is added for null.qubit tracking resources and depth."""
        dev = qp.device(
            "null.qubit",
            wires=2,
            track_resources=True,
            resources_filename=str(tmp_path / "resources.json"),
            compute_depth=True,
        )

        @qjit
        @qp.qnode(dev)
        def circuit():
            qp.cond(qp.measure(0), qp.Z)(1)
            return qp.probs(wires=1)

        circuit()
        passes = _bufferization_passes(circuit)
        assert passes[0] == CLASSICAL_CONTROL_PASS
        assert passes.count(CLASSICAL_CONTROL_PASS) == 1

    @pytest.mark.parametrize(
        "device_kwargs",
        [
            {},
            {"track_resources": True},
            {"track_resources": True, "compute_depth": False},
        ],
    )
    def test_not_tracking_depth(self, device_kwargs, tmp_path):
        """Test that the pass is not added for null.qubit without depth tracking."""
        if device_kwargs.get("track_resources"):
            device_kwargs["resources_filename"] = str(tmp_path / "resources.json")
        dev = qp.device("null.qubit", wires=2, **device_kwargs)

        @qjit
        @qp.qnode(dev)
        def circuit():
            qp.cond(qp.measure(0), qp.Z)(1)
            return qp.probs(wires=1)

        circuit()
        assert CLASSICAL_CONTROL_PASS not in _bufferization_passes(circuit)

    def test_other_device(self, backend):
        """Test that the pass is not added for devices other than null.qubit."""

        @qjit
        @qp.qnode(qp.device(backend, wires=2))
        def circuit():
            qp.cond(qp.measure(0), qp.Z)(1)
            return qp.probs(wires=1)

        circuit()
        assert CLASSICAL_CONTROL_PASS not in _bufferization_passes(circuit)


class TestDepth:
    """Test the circuit depth tracked for classically controlled operations."""

    def test_cond(self, capture_mode, tmp_path):
        """Test that a conditional operation is placed after the measurement it depends on."""

        def body():
            m = qp.measure(0, postselect=1)
            qp.cond(m, qp.Z)(1)

        resources = _track(body, capture_mode, tmp_path)
        assert resources["gate_types"] == {"PauliZ": 1}
        assert resources["depth"] == 2

    def test_cond_else_branch(self, capture_mode, tmp_path):
        """Test that the else branch of a conditional is placed after the measurement."""

        def body():
            m = qp.measure(0)
            qp.cond(m, qp.X, qp.Z)(1)

        resources = _track(body, capture_mode, tmp_path)
        assert resources["gate_types"] == {"PauliZ": 1}
        assert resources["depth"] == 2

    def test_unconditional_operation(self, capture_mode, tmp_path):
        """Test that operations outside the conditional are not placed after the measurement."""

        def body():
            m = qp.measure(0, postselect=1)
            qp.cond(m, qp.Z)(1)
            qp.X(2)
            qp.X(2)

        resources = _track(body, capture_mode, tmp_path)
        assert resources["depth"] == 2

    def test_mid_circuit_measurement(self, capture_mode, tmp_path):
        """Test that a mid-circuit measurement occupies a layer on its wire."""

        def body():
            qp.Hadamard(0)
            qp.measure(0)

        resources = _track(body, capture_mode, tmp_path, wires=1)
        assert resources["measurements"]["MidMeasure"] == 1
        assert resources["depth"] == 2

    def test_nested_cond(self, capture_mode, tmp_path):
        """Test that nested conditionals are placed after the measurements they depend on."""

        def body():
            m0 = qp.measure(0, postselect=1)

            def branch():
                m1 = qp.measure(1, postselect=1)
                qp.cond(m1, qp.Z)(2)

            qp.cond(m0, branch)()

        resources = _track(body, capture_mode, tmp_path)
        assert resources["measurements"]["MidMeasure"] == 2
        assert resources["depth"] == 3

    def test_repeat_until_success(self, capture_mode, tmp_path):
        """Test that the body of a loop conditioned on measurements is placed after them."""

        def body():
            m0 = qp.measure(0, postselect=1)

            @qp.while_loop(lambda m: m)
            def loop(m):
                qp.X(1)
                return qp.measure(2)

            loop(m0)

        resources = _track(body, capture_mode, tmp_path)
        assert resources["gate_types"] == {"PauliX": 1}
        assert resources["depth"] == 2


if __name__ == "__main__":
    pytest.main(["-x", __file__])
