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

"""Exact analysis of deterministic runtime synthesis, without sampling mixtures."""

import json

import numpy as np
import pennylane as qp
import pytest

from catalyst import qjit
from catalyst.compiler import _catalyst
from catalyst.passes import to_ppr
from catalyst.utils.pbc_classical import resource_analysis_options
from catalyst.utils.runtime_environment import INSTALLED


def _rotation_circuit(method, shared_wire=False):
    epsilon = 1e-4
    costs = {
        "PPR_8": 100,
        "PPR_4": 10,
        "PPR_2": 1,
        "GlobalPhase": 0,
        "PauliMeasure": 2,
        "PauliMeasureMP": 2,
        "MidMeasure": 2,
        "MidMeasureMP": 2,
        "MidCircuitMeasure": 2,
        "RZ": 110 * (0.58 * np.log2(1 / epsilon) + 3.7),
    }

    @qjit(capture=True, target="mlir", pipelines=[("pipe", ["quantum-compilation-stage"])])
    @to_ppr
    @qp.transforms.gridsynth(method=method, epsilon=epsilon, ppr_basis=True)
    @qp.transform(pass_name="symbol-dce")
    @qp.decompose(gate_set=costs)
    @qp.qnode(qp.device("null.qubit", wires=2))
    def circuit():
        qp.RX(0.2, wires=0)
        qp.RY(0.2, wires=0 if shared_wire else 1)

    return circuit


@pytest.mark.parametrize("shared_wire, expected", [(False, 44), (True, 86)])
def test_deterministic_rotation_depth(shared_wire, expected, monkeypatch):
    """The actual 40-PPR sequences preserve parallelism and their basis changes."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    resources = qp.analyze(
        _rotation_circuit("deterministic", shared_wire), level="user"
    )().resources
    assert resources.extra["qubit_disjoint_depth_status"] == "exact"
    assert resources.qubit_disjoint_depth == expected


def test_mixed_fallback_remains_unavailable():
    """A runtime-random sequence length must not become an exact expected depth."""
    with pytest.warns(UserWarning, match="Exact qubit-disjoint depth is unavailable"):
        resources = qp.analyze(_rotation_circuit("mixed-fallback"), level="user")().resources
    assert resources.extra["qubit_disjoint_depth_status"] == "unavailable"
    assert resources.qubit_disjoint_depth is None


def _analyze_ir(source, load_runtime=True):
    options = resource_analysis_options() if load_runtime else {}
    pipeline = (
        "builtin.module(resource-analysis{output-json "
        + " ".join(f"{name.replace('_', '-')}={value}" for name, value in options.items())
        + "})"
    )
    output = _catalyst(
        ("--tool", "opt"), ("--pass-pipeline", pipeline), ("-o", "/dev/null"), stdin=source
    )
    return json.loads(output)["bench"]["extended_fields"]["pbc_depth"]


def _buffer_program(extra_use="", angle="%theta", allocate_size="%size"):
    return f"""module {{
      func.func private @rs_decomposition_get_size(f64, f64, i1) -> index
      func.func private @rs_decomposition_get_gates(memref<?xindex>, f64, f64, i1)
      func.func @bench(%q: !quantum.bit, %unknown: f64) {{
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %theta = arith.constant 1.5707963267948966 : f64
        %epsilon = arith.constant 1.0e-4 : f64
        %true = arith.constant true
        %size = func.call @rs_decomposition_get_size({angle}, %epsilon, %true)
            : (f64, f64, i1) -> index
        %buffer = memref.alloc({allocate_size}) : memref<?xindex>
        func.call @rs_decomposition_get_gates(%buffer, {angle}, %epsilon, %true)
            : (memref<?xindex>, f64, f64, i1) -> ()
        {extra_use}
        %gate = memref.load %buffer[%zero] : memref<?xindex>
        %expected = arith.constant 14 : index
        %correct = arith.cmpi eq, %gate, %expected : index
        %out = scf.if %correct -> (!quantum.bit) {{
          %r = pbc.ppr ["Z"](4) %q : !quantum.bit
          scf.yield %r : !quantum.bit
        }} else {{
          %r = pbc.ppr ["X"](8) %q : !quantum.bit
          %s = pbc.ppr ["X"](8) %r : !quantum.bit
          scf.yield %s : !quantum.bit
        }}
        memref.dealloc %buffer : memref<?xindex>
        return
      }}
    }}"""


def test_deterministic_gate_buffer():
    """Known synthesis values choose the actual gate-selector branch."""
    depth = _analyze_ir(_buffer_program())
    assert depth["qubit_disjoint_depth_status"] == "exact"
    assert depth["qubit_disjoint_depth_total"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "memref.store %zero, %buffer[%zero] : memref<?xindex>",
        "%alias = memref.cast %buffer : memref<?xindex> to memref<1xindex>",
        "memref.dealloc %buffer : memref<?xindex>",
    ],
)
def test_modified_gate_buffer_is_not_certified(mutation):
    """Mutated, aliased, and prematurely released buffers are not read as constants."""
    depth = _analyze_ir(_buffer_program(extra_use=mutation))
    assert depth["qubit_disjoint_depth_status"] == "unavailable"
    assert "qubit_disjoint_depth_total" not in depth


def test_unknown_synthesis_angle_is_not_certified():
    depth = _analyze_ir(_buffer_program(angle="%unknown"))
    assert depth["qubit_disjoint_depth_status"] == "unavailable"
    assert "qubit_disjoint_depth_total" not in depth


def test_undersized_gate_buffer_is_not_certified():
    depth = _analyze_ir(_buffer_program(allocate_size="%zero"))
    assert depth["qubit_disjoint_depth_status"] == "unavailable"
    assert "qubit_disjoint_depth_total" not in depth


def test_gate_buffer_read_before_fill_is_not_certified():
    source = _buffer_program()
    read = "%gate = memref.load %buffer[%zero] : memref<?xindex>"
    source = source.replace(read, "")
    source = source.replace(
        "func.call @rs_decomposition_get_gates(%buffer",
        read + "\n        func.call @rs_decomposition_get_gates(%buffer",
    )
    depth = _analyze_ir(source)
    assert depth["qubit_disjoint_depth_status"] == "unavailable"
    assert "qubit_disjoint_depth_total" not in depth


def test_missing_synthesis_runtime_is_not_certified():
    depth = _analyze_ir(_buffer_program(), load_runtime=False)
    assert depth["qubit_disjoint_depth_status"] == "unavailable"
    assert "qubit_disjoint_depth_total" not in depth


@pytest.mark.skipif(INSTALLED, reason="Runtime path overrides apply to source builds")
def test_synthesis_runtime_directory_override(monkeypatch, tmp_path):
    """Analysis resolves synthesis with the same runtime override as compilation."""
    monkeypatch.setenv("RUNTIME_LIB_DIR", str(tmp_path / "missing-runtime"))
    depth = _analyze_ir(_buffer_program())
    assert depth["qubit_disjoint_depth_status"] == "unavailable"
    assert "qubit_disjoint_depth_total" not in depth
