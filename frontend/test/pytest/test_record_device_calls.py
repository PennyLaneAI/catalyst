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
"""Tests for capture frontend resource recording via RecordDeviceCalls."""

import json

import pennylane as qp
import pytest

from catalyst import qjit
from catalyst.device import RecordDeviceCalls, extract_backend_info
from catalyst.from_plxpr.from_plxpr import _get_device_kwargs

pytestmark = pytest.mark.usefixtures("use_capture")


def test_record_path_must_be_nonempty():
    """Empty recording paths are rejected."""
    with pytest.raises(ValueError, match="non-empty"):
        RecordDeviceCalls(qp.device("null.qubit", wires=1), "")


def test_wrapper_forwards_device_identity():
    """The wrapper exposes the inner device name and wires."""
    inner = qp.device("null.qubit", wires=2)
    wrapped = RecordDeviceCalls(inner, "calls.jsonl")
    assert wrapped.name == inner.name
    assert wrapped.wires == inner.wires
    assert wrapped.original_device is inner
    assert wrapped.record_path == "calls.jsonl"
    assert wrapped.device_kwargs["record_device_calls"] == "calls.jsonl"


def test_extract_backend_info_injects_recording_path():
    """extract_backend_info adds the recording path to runtime kwargs."""
    inner = qp.device("null.qubit", wires=1)
    wrapped = RecordDeviceCalls(inner, "calls with,punctuation.jsonl")

    raw_info = extract_backend_info(inner)
    wrapped_info = extract_backend_info(wrapped)

    assert raw_info.c_interface_name == wrapped_info.c_interface_name
    assert "record_device_calls" not in raw_info.kwargs
    assert wrapped_info.kwargs["record_device_calls"] == "calls%20with%2Cpunctuation.jsonl"

    kwargs = _get_device_kwargs(wrapped)
    assert "record_device_calls" in kwargs["rtd_kwargs"]


def test_unwrapped_device_does_not_create_log(tmp_path):
    """A raw device does not write a recording file."""
    log = tmp_path / "unused.jsonl"
    dev = qp.device("null.qubit", wires=1)

    @qjit(capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.expval(qp.PauliZ(0))

    circuit()
    assert not log.exists()


def test_wrapped_device_records_runtime_calls(tmp_path):
    """A wrapped device records resources using NullQubit's output format."""
    log = tmp_path / "resources.json"
    dev = RecordDeviceCalls(qp.device("null.qubit", wires=1), str(log))

    @qjit(capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.Hadamard(0)
        return qp.expval(qp.PauliZ(0))

    circuit()

    assert log.exists()
    resources = json.loads(log.read_text(encoding="utf-8"))
    assert resources["num_wires"] == 1
    assert resources["num_gates"] == 1
    assert resources["gate_types"]["Hadamard"] == 1
    assert resources["measurements"]["expval(PauliZ)"] == 1


def test_existing_record_file_is_replaced(tmp_path):
    """Recording over an existing file overwrites it instead of aborting the process."""
    log = tmp_path / "resources.json"
    log.write_text("stale contents", encoding="utf-8")

    dev = RecordDeviceCalls(qp.device("null.qubit", wires=1), str(log))

    @qjit(capture=True)
    @qp.qnode(dev)
    def circuit():
        qp.RZ(0.5, 0)
        return qp.expval(qp.PauliZ(0))

    circuit()

    resources = json.loads(log.read_text(encoding="utf-8"))
    assert resources["gate_types"]["RZ"] == 1
