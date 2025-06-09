# Copyright 2025 Xanadu Quantum Technologies Inc.

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
This file contains tests for automatic qubit management.
Automatic qubit management refers to when the user does not specify the total number of wires
during device initialization.

Note that, catalyst and pennylane handles device labels differently. For example, when a new label
    qml.gate_or_measurement(wires=1000)
is encountered, core pennylane considers "1000" as a pure wire label, and interprets that as
*one* new wire, with the label "1000". However in catalyst we do not associate wires with
arbitrary labels and require wires to be continuous integers from zero, and we would interpret this
as "allocate new wires until we have 1001 wires, and act on wire[1000]".

In other words, the reference runs of automatic qubit management should be qjit runs with wires
specified during device initialization, instead of non qjit runs.
"""

import numpy as np
import pennylane as qml

from catalyst import qjit


def test_partial_sample(backend):
    """
    Test that a `sample` terminal measurement with wires specified can be executed
    correctly with automatic qubit management.
    """

    def circuit():
        qml.RX(0.0, wires=0)
        return qml.sample(wires=[0, 2])

    wires = [4, None]
    devices = [qml.device(backend, shots=10, wires=wire) for wire in wires]
    ref, observed = (qjit(qml.qnode(dev)(circuit))() for dev in devices)
    assert ref.shape == observed.shape
    assert np.allclose(ref, observed)


def test_partial_counts(backend):
    """
    Test that a `counts` terminal measurement with wires specified can be executed
    correctly with automatic qubit management.
    """

    def circuit():
        qml.RX(0.0, wires=0)
        return qml.counts(wires=[0, 2])

    wires = [4, None]
    devices = [qml.device(backend, shots=10, wires=wire) for wire in wires]
    ref, observed = (qjit(qml.qnode(dev)(circuit))() for dev in devices)
    assert (ref[i].shape == observed[i].shape for i in (0, 1))
    assert np.allclose(ref, observed)


def test_partial_probs(backend):
    """
    Test that a `probs` terminal measurement with wires specified can be executed
    correctly with automatic qubit management.
    """

    def circuit():
        qml.PauliX(wires=0)
        return qml.probs(wires=[0, 2])

    wires = [4, None]
    devices = [qml.device(backend, wires=wire) for wire in wires]
    ref, observed = (qjit(qml.qnode(dev)(circuit))() for dev in devices)
    assert ref.shape == observed.shape
    assert np.allclose(ref, observed)


def test_sample(backend):
    """
    Test that a `sample` terminal measurement without wires specified can be executed
    correctly with automatic qubit management.
    """

    def circuit():
        qml.RX(0.0, wires=3)
        return qml.sample()

    wires = [4, None]
    devices = [qml.device(backend, shots=10, wires=wire) for wire in wires]
    ref, observed = (qjit(qml.qnode(dev)(circuit))() for dev in devices)
    assert ref.shape == observed.shape
    assert np.allclose(ref, observed)


def test_counts(backend):
    """
    Test that a `counts` terminal measurement without wires specified can be executed
    correctly with automatic qubit management.
    """

    def circuit():
        qml.RX(0.0, wires=3)
        return qml.counts()

    wires = [4, None]
    devices = [qml.device(backend, shots=10, wires=wire) for wire in wires]
    ref, observed = (qjit(qml.qnode(dev)(circuit))() for dev in devices)
    assert (ref[i].shape == observed[i].shape for i in (0, 1))
    assert np.allclose(ref, observed)


def test_probs(backend):
    """
    Test that a `probs` terminal measurement without wires specified can be executed
    correctly with automatic qubit management.
    """

    def circuit():
        qml.PauliX(wires=3)
        return qml.probs()

    wires = [4, None]
    devices = [qml.device(backend, wires=wire) for wire in wires]
    ref, observed = (qjit(qml.qnode(dev)(circuit))() for dev in devices)
    assert ref.shape == observed.shape
    assert np.allclose(ref, observed)


def test_state(backend):
    """
    Test that a `state` terminal measurement without wires specified can be executed
    correctly with automatic qubit management.
    """

    def circuit():
        qml.PauliX(wires=3)
        return qml.state()

    wires = [4, None]
    devices = [qml.device(backend, wires=wire) for wire in wires]
    ref, observed = (qjit(qml.qnode(dev)(circuit))() for dev in devices)
    assert ref.shape == observed.shape
    assert np.allclose(ref, observed)
