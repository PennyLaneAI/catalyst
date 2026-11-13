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
from catalyst.debug import get_compilation_stage

TAG_PASS = "tag-classical-control-measurements"
INSTRUMENT_PASS = "instrument-classical-control"


def _null_qubit(tmp_path, wires=3, **device_kwargs):
    """Return a null.qubit device tracking resources with ``device_kwargs``."""
    if device_kwargs.get("track_resources"):
        device_kwargs.setdefault("resources_filename", str(tmp_path / "resources.json"))
    return qp.device("null.qubit", wires=wires, **device_kwargs)


def _track(body, tmp_path, wires=3):
    """Execute ``body`` on null.qubit with depth tracking, returning the tracked resources."""
    dev = _null_qubit(tmp_path, wires=wires, track_resources=True, compute_depth=True)

    @qjit(capture=True)
    @qp.qnode(dev)
    def circuit():
        body()
        return qp.probs(wires=0)

    circuit()
    with (tmp_path / "resources.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _cond_on_measurement():
    """Apply an operation conditioned on a mid-circuit measurement."""
    qp.cond(qp.measure(0), qp.Z)(1)
    return qp.probs(wires=1)


class TestPipeline:
    """Test that only the measurements of QNodes tracking the circuit depth are instrumented."""

    def test_instrumentation_in_bufferization_stage(self):
        """Test that the instrumentation pass is the first pass of the bufferization stage."""

        @qjit(capture=True)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def circuit():
            return _cond_on_measurement()

        passes = dict(circuit.compile_options.get_pipelines())["BufferizationStage"]
        assert passes[0] == INSTRUMENT_PASS

    def test_tracking_depth(self, tmp_path):
        """Test that device preprocessing tags the measurements of null.qubit tracking resources
        and depth."""
        dev = _null_qubit(tmp_path, wires=2, track_resources=True, compute_depth=True)

        @qjit(capture=True)
        @qp.qnode(dev)
        def circuit():
            return _cond_on_measurement()

        assert f'"{TAG_PASS}"' in circuit.mlir

    @pytest.mark.parametrize(
        "device_kwargs",
        [
            {},
            {"track_resources": True},
            {"track_resources": True, "compute_depth": False},
        ],
    )
    def test_not_tracking_depth(self, device_kwargs, tmp_path):
        """Test that device preprocessing does not tag the measurements of null.qubit without
        depth tracking."""

        @qjit(capture=True)
        @qp.qnode(_null_qubit(tmp_path, wires=2, **device_kwargs))
        def circuit():
            return _cond_on_measurement()

        assert TAG_PASS not in circuit.mlir

    def test_other_device(self, backend):
        """Test that device preprocessing does not tag the measurements of other devices."""

        @qjit(capture=True)
        @qp.qnode(qp.device(backend, wires=2))
        def circuit():
            return _cond_on_measurement()

        assert TAG_PASS not in circuit.mlir

    def test_only_tracking_qnodes_instrumented(self, backend, tmp_path, monkeypatch):
        """Test that only the QNodes tracking the circuit depth are instrumented in a program
        with QNodes on different devices."""
        monkeypatch.chdir(tmp_path)
        dev = _null_qubit(tmp_path, wires=2, track_resources=True, compute_depth=True)

        @qp.qnode(dev)
        def tracked():
            return _cond_on_measurement()

        @qp.qnode(qp.device(backend, wires=2))
        def untracked():
            return _cond_on_measurement()

        @qjit(capture=True, keep_intermediate=True)
        def workflow():
            return tracked(), untracked()

        workflow()
        ir = get_compilation_stage(workflow, "BufferizationStage")
        assert ir.count("call @__catalyst__rt__begin_classical_control") == 1
        assert "quantum.track_classical_control" not in ir


class TestDepth:
    """Test the circuit depth tracked for classically controlled operations."""

    def test_cond(self, tmp_path):
        """Test that a conditional operation is placed after the measurement it depends on."""

        def body():
            m = qp.measure(0, postselect=1)
            qp.cond(m, qp.Z)(1)

        resources = _track(body, tmp_path)
        assert resources["gate_types"] == {"PauliZ": 1}
        assert resources["depth"] == 2

    def test_cond_else_branch(self, tmp_path):
        """Test that the else branch of a conditional is placed after the measurement."""

        def body():
            m = qp.measure(0)
            qp.cond(m, qp.X, qp.Z)(1)

        resources = _track(body, tmp_path)
        assert resources["gate_types"] == {"PauliZ": 1}
        assert resources["depth"] == 2

    def test_unconditional_operation(self, tmp_path):
        """Test that operations outside the conditional are not placed after the measurement."""

        def body():
            m = qp.measure(0, postselect=1)
            qp.cond(m, qp.Z)(1)
            qp.X(2)
            qp.X(2)

        resources = _track(body, tmp_path)
        assert resources["depth"] == 2

    def test_mid_circuit_measurement(self, tmp_path):
        """Test that a mid-circuit measurement occupies a layer on its wire."""

        def body():
            qp.Hadamard(0)
            qp.measure(0)

        resources = _track(body, tmp_path, wires=1)
        assert resources["measurements"]["MidMeasure"] == 1
        assert resources["depth"] == 2

    def test_nested_cond(self, tmp_path):
        """Test that nested conditionals are placed after the measurements they depend on."""

        def body():
            m0 = qp.measure(0, postselect=1)

            def branch():
                m1 = qp.measure(1, postselect=1)
                qp.cond(m1, qp.Z)(2)

            qp.cond(m0, branch)()

        resources = _track(body, tmp_path)
        assert resources["measurements"]["MidMeasure"] == 2
        assert resources["depth"] == 3

    def test_repeat_until_success(self, tmp_path):
        """Test that the body of a loop conditioned on measurements is placed after them."""

        def body():
            m0 = qp.measure(0, postselect=1)

            @qp.while_loop(lambda m: m)
            def loop(m):
                qp.X(1)
                return qp.measure(2)

            loop(m0)

        resources = _track(body, tmp_path)
        assert resources["gate_types"] == {"PauliX": 1}
        assert resources["depth"] == 2


if __name__ == "__main__":
    pytest.main(["-x", __file__])
