# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from catalyst import qjit, measure
import pennylane as qml
import jax.numpy as jnp
import jax

from catalyst.jax_tracer import get_traceable_fn


class TestMidCircuitMeasurementsToJax:
    def test_simple_mid_circuit_measurement(self):
        expected = """{ lambda ; a:f64[]. let
    b:AbstractQreg() = qalloc 2
    c:AbstractQbit() = qextract b 0
    d:AbstractQbit() = qinst[op=RX qubits_len=1] c a
    e:bool[] f:AbstractQbit() = qmeasure d
    _:AbstractQbit() = qinst[op=RY qubits_len=1] f e
    g:AbstractQbit() = qextract b 1
    h:bool[] _:AbstractQbit() = qmeasure g
     = qdealloc b
  in (h,) }"""

        def circuit(x):
            qml.RX(x, wires=0)
            m = measure(wires=0)
            qml.RY(m, wires=0)
            return measure(wires=1)

        device = qml.device("lightning.qubit", wires=2, shots=1)
        traceable_fn = get_traceable_fn(circuit, device)

        assert expected == str(jax.make_jaxpr(traceable_fn)(1.0))

    def test_multiply_mid_circuit_measurement(self):
        expected = """{ lambda ; a:f64[]. let
    b:AbstractQreg() = qalloc 2
    c:AbstractQbit() = qextract b 0
    d:AbstractQbit() = qinst[op=RX qubits_len=1] c a
    e:bool[] f:AbstractQbit() = qmeasure d
    g:f64[] = convert_element_type[new_dtype=float64 weak_type=True] e
    h:f64[] = mul 2.0 g
    _:AbstractQbit() = qinst[op=RY qubits_len=1] f h
    i:AbstractQbit() = qextract b 1
    j:bool[] _:AbstractQbit() = qmeasure i
     = qdealloc b
  in (j,) }"""

        def circuit(x):
            qml.RX(x, wires=0)
            m = measure(wires=0)
            qml.RY(2.0 * m, wires=0)
            return measure(wires=1)

        device = qml.device("lightning.qubit", wires=2, shots=1)
        traceable_fn = get_traceable_fn(circuit, device)

        assert expected == str(jax.make_jaxpr(traceable_fn)(1.0))


class TestMidCircuitMeasurement:
    def test_pl_measure(self):
        def circuit():
            return qml.measure(0)

        with pytest.raises(TypeError, match="Must use 'measure' from Catalyst"):
            qjit(qml.qnode(qml.device("lightning.qubit", wires=1))(circuit))()

    def test_basic(self):
        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=1))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m = measure(wires=0)
            return m

        assert circuit(jnp.pi)  # m will be equal to True if wire 0 is measured in 1 state

    def test_more_complex(self):
        @qjit()
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: float):
            qml.RX(x, wires=0)
            m1 = measure(wires=0)
            maybe_pi = m1 * jnp.pi
            qml.RX(maybe_pi, wires=1)
            m2 = measure(wires=1)
            return m2

        assert circuit(jnp.pi)  # m will be equal to True if wire 0 is measured in 1 state
        assert not circuit(0.0)


if __name__ == "__main__":
    pytest.main(["-x", __file__])
