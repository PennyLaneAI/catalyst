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
Helpers for the from_plxpr conversion.
"""

from catalyst.jax_primitives import (
    AbstractQbit,
    AbstractQreg,
    qalloc_p,
    qdealloc_p,
    qextract_p,
    qinsert_p,
)


class QregManager:
    """
    A manager that handles converting plxpr wire indices into catalyst jaxpr qreg.

    The fundamental challenge in from_plxpr is that Plxpr (and frontend PennyLane) uses wire index
    semantics, but Catalyst jaxpr uses qubit value semantics.

    In plxpr, there is the notion of an implicit global state, and each operation (gates, or meta-
    ops like control flow or adjoint) is essentially an update to the global state. At any time,
    the target of an operation is the implicit global state. This is also in line with devices:
    most simulators have a global state vector, and hardware has the one piece of hardware that
    gates are applied to.

    However, in Catalyst no such global state exists. Each (pure) operation consumes an SSA qubit
    (or qreg for meta ops) value and return a new qubit (or qreg) value. The target of an operation,
    while expressed by a plain wire index in plxpr, needs to be converted to the **current**
    SSA qubit value at that index in catalyst jaxpr.

    For example, the RX operation in the snippet (pseudo code)
        qreg:AbstractQreg() = ...
        q0:AbstractQbit() = qextract qreg 0
        q1:AbstractQbit() = qinst[
           op=RX
        ] q0 3.14
    would update the current qubit value on index 0 of register qreg from q0 to q1.

    The same goes for registers. The for loop operation in this snippet (pseudo code)
        qreg0:AbstractQreg() = qalloc 3
        qreg1:AbstractQreg() = for_loop[
            body_jaxpr={ lambda ; d:i64[] e:AbstractQreg(). let
                ...
                yield:AbstractQreg() = ...
              in (yield,) }
        ] ...for_loop_args... qreg0
    would update the current qreg value for that particular allocation from qreg0 to qreg1.

    There are two places where a fresh, new qreg value (i.e. the root qreg value of a def-use
    chain) can appear in catalyst jaxpr:
    1. From a brand new allocation, i.e. the result of a qalloc primitive.
    2. From an argument of the current scope's jaxpr.

    With the above in mind, this `QregManager` class promises the following:
    1. An instance of this class will always be tied to one root qreg value.
    2. At any moment during the from_plxpr conversion:
       - `QregManager.get()` returns the current catalyst qreg SSA value for the managed
          root register on that instance;
       - `QregManager[i]` returns the current catalyst qubit SSA value for the i-th index
          on the managed root register on that instance. If none exists, a new qubit will be
          extracted.

    To achieve the above, users of this class are expected to:
    1. Initialize an instance with the qreg SSA value from a new allocation or a new block argument;
    2. Whenever a new meta-op/op is bind-ed, update the current qreg/qubit SSA value with:
       - `QubitManager.set(new_qreg_value)`
       - `QubitManager[i] = new_qubit_value`
    """

    wire_map: dict[int, AbstractQbit]  # Note: No dynamic wire indices for now in from_plxpr.

    def __init__(self, root_qreg_value: AbstractQreg):
        self.abstract_qreg_val = root_qreg_value
        self.wire_map = {}

    def get(self):
        """Return the current AbstractQreg value."""
        return self.abstract_qreg_val

    def set(self, qreg: AbstractQreg):
        """
        Set the current AbstractQreg value.
        This is needed, for example, when existing regions, like submodules and control flow.
        """
        self.abstract_qreg_val = qreg

    def extract(self, index: int) -> AbstractQbit:
        """Create the extract primitive that produces an AbstractQbit value."""

        # extract must be fresh
        assert index not in self.wire_map
        extracted_qubit = qextract_p.bind(self.abstract_qreg_val, index)
        self.wire_map[index] = extracted_qubit
        return extracted_qubit

    def insert(self, index: int, qubit: AbstractQbit):
        """
        Create the insert primitive.
        """
        self.abstract_qreg_val = qinsert_p.bind(self.abstract_qreg_val, index, qubit)
        self.wire_map.pop(index)

    def insert_all_dangling_qubits(self):
        """
        Insert all dangling qubits back into a qreg.

        This is necessary, for example, at the end of the qreg lifetime before deallocing,
        or when passing qregs into and out of scopes like control flow.
        """
        for index, qubit in self.wire_map.items():
            self.abstract_qreg_val = qinsert_p.bind(self.abstract_qreg_val, index, qubit)
        self.wire_map.clear()

    def __getitem__(self, index: int) -> AbstractQbit:
        """
        Get the newest ``AbstractQbit`` corresponding to a wire index.
        If the qubit value does not exist yet at this index, extract the fresh qubit.
        """
        if index in self.wire_map:
            return self.wire_map[index]
        return self.extract(index)

    def __setitem__(self, index: int, qubit: AbstractQbit):
        """
        Update the wire_map when a new qubit value for a wire index is produced,
        for example by gates.
        """
        self.wire_map[index] = qubit
