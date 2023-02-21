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

from catalyst import measure
import pennylane as qml

from catalyst.jax_tape import JaxTape


class TestJaxTape:
    def test_simple_jax_tape(self):
        parameter_fn = "{ lambda ; . let  in (1.5, 0, 0) }"

        with JaxTape(do_queue=False) as tape:
            with tape.quantum_tape:
                qml.RX(1.5, wires=0)
                qml.sample(qml.PauliZ(wires=0))

        assert str(tape.closed_jaxpr) == parameter_fn

    def test_measurement_jax_tape(self):
        parameter_fn = "{ lambda ; a:bool[]. let  in (1.5, 0, 0, a, 0, 0) }"

        with JaxTape(do_queue=False) as tape:
            with tape.quantum_tape:
                qml.RX(1.5, wires=0)
                m = measure(wires=0)
                qml.RY(m, wires=0)
                qml.sample(qml.PauliZ(wires=0))

        assert str(tape.closed_jaxpr) == parameter_fn

    def test_multiple_measurement_jax_tape(self):
        parameter_fn = """{ lambda ; a:bool[] b:bool[]. let
    c:f64[] = convert_element_type[new_dtype=float64 weak_type=True] a
    d:f64[] = add c 0.0
    e:f64[] = convert_element_type[new_dtype=float64 weak_type=True] b
    f:f64[] = add d e
  in (1.5, 0, 0, d, 0, 0, f, 0, 0) }"""

        with JaxTape(do_queue=False) as tape:
            with tape.quantum_tape:
                qml.RX(1.5, wires=0)
                m1 = measure(wires=0) + 0.0  # need this to force cast to float
                qml.RY(m1, wires=0)
                m2 = measure(wires=0)
                qml.RY(m1 + m2, wires=0)
                qml.sample(qml.PauliZ(wires=0))

        assert str(tape.closed_jaxpr) == parameter_fn


if __name__ == "__main__":
    pytest.main(["-x", __file__])
