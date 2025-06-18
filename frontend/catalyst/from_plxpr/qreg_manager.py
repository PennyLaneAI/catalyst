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
Qubit value managers for from_plxpr conversion.

The fundamental challenge in from_plxpr is that Plxpr (and frontend PennyLane) uses wire index
semantics, but Catalyst jaxpr uses qubit value semantics.

In plxpr, there is the notion of an implicit global state, and each operation (gates, or meta-
ops like control flow, adjoint, or subroutines) is essentially an update to the global state.
At any time, the target of an operation is the implicit global state. This is also in line with
devices: most simulators have a global state vector, and hardware has the one piece of hardware
that gates are applied to.

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

Importantly, registers are never explicit in plxpr. Plxpr only maintains a list of global wire
indices (integer for static, int tracer for dynamic), and all operations are explicitly targeting
the wire indices. The notion of a "register" is maintained by groups of indices in plxpr.
For example, the `qml.allocate()` would trace to the following primitive:
    b:i64[] c:i64[] d:i64[] = allocate[
        num_wires=3
        other_params=...
    ]
and the integer indices `b, c, d` are logically a register, corresponding to a catalyst qreg value.

Therefore, the two central questions we wish to answer is:
1. Given a plain wire index in plxpr, which logical catalyst register does this wire belong to?
2. Given a logical catalyst register, what is its current qreg SSA value, and what are its current
   qubit SSA values on its wires?
"""

from catalyst.jax_primitives import AbstractQbit, AbstractQreg, qextract_p, qinsert_p


class QubitIndexRecorder:
    """
    A manager that records what wire index belongs to what logical register.

    This manager answers the first question.
    Since plxpr indices are global, there should be one instance of the class per plxpr.

    This class promises the following:
    - `QubitIndexRecorder[i]` (i = global index in plxpr) returns the `QRegManager` instance
       for the logical register that this wire belongs to.

    Users of this class are expected to:
    - Create an instance of this class per plxpr
    - Pass that instance to all `QregManager` instances created when tracing that plxpr
    """

    def __init__(self):
        self.map = {}

    def __getitem__(self, index: int):  # -> QregManager
        assert index in self.map
        return self.map[index]

    def __setitem__(self, index: int, qreg):
        # `qreg` is of type `QregManager`
        self.map[index] = qreg


class QregManager:
    """
    A manager that handles converting plxpr wire indices into catalyst jaxpr qreg.

    This manager answers the second question, i.e. given a logical catalyst register,
    what is its current qreg SSA value, and what are its current qubit SSA values on its wires?

    This `QregManager` class promises the following:
    1. An instance of this class will always be tied to one root qreg value.
    2. At any moment during the from_plxpr conversion:
       - `QregManager.get()` returns the current catalyst qreg SSA value for the managed
          root register on that instance;
       - `QregManager[i]` (i=0,1,2....) returns the current catalyst qubit SSA value for the i-th
          index on the managed root register on that instance. If none exists, a new qubit will be
          extracted.

    To achieve the above, users of this class are expected to:
    1. Initialize an instance with the qreg SSA value from a new allocation or a new block argument;
    2. Whenever a new meta-op/op is bind-ed, update the current qreg/qubit SSA value with:
       - `QubitManager.set(new_qreg_value)`
       - `QubitManager[i] = new_qubit_value` (i=0,1,2....)
    """

    wire_map: dict[int, AbstractQbit]  # Note: No dynamic wire indices for now in from_plxpr.

    def __init__(self, root_qreg_value: AbstractQreg, recorder: QubitIndexRecorder):
        self.current_qreg_value = root_qreg_value
        self.recorder = recorder

        # A map from plxpr's *global* indices to the current catalyst SSA qubit values
        # for the wires on this qreg.
        self.wire_map = {}

        # Plxpr works with absolute indices, i.e. all unique, even across registers
        # However, catalyst qreg work with relative indices
        # i.e. extracting and inserting with every new catalyst qreg use indices 0,1,2,...
        # To distinguish between different registers, we require that each one has a hash.
        self.root_hash = hash(self.current_qreg_value)

    def get(self):
        """Return the current AbstractQreg value."""
        return self.current_qreg_value

    def set(self, qreg: AbstractQreg):
        """
        Set the current AbstractQreg value.
        This is needed, for example, when existing regions, like submodules and control flow.
        """
        self.current_qreg_value = qreg

    def extract(self, index: int) -> AbstractQbit:
        """Create the extract primitive that produces an AbstractQbit value."""

        global_index = self.local_index_to_global_index(index)

        # extract must be fresh
        assert global_index not in self.wire_map

        # record that this global wire index is on this register
        self.recorder[global_index] = self

        # extract and update current qubit value
        extracted_qubit = qextract_p.bind(self.current_qreg_value, index)
        self.wire_map[global_index] = extracted_qubit
        return extracted_qubit

    def insert(self, index: int, qubit: AbstractQbit):
        """
        Create the insert primitive.
        """
        global_index = self.local_index_to_global_index(index)
        self.current_qreg_value = qinsert_p.bind(self.current_qreg_value, index, qubit)
        self.wire_map.pop(global_index)

    def insert_all_dangling_qubits(self):
        """
        Insert all dangling qubits back into a qreg.

        This is necessary, for example, at the end of the qreg lifetime before deallocing,
        or when passing qregs into and out of scopes like control flow.
        """
        for global_index, qubit in self.wire_map.items():
            self.current_qreg_value = qinsert_p.bind(
                self.current_qreg_value, global_index - self.root_hash, qubit
            )
        self.wire_map.clear()

    def get_all_current_global_indices(self):
        """
        Return a list of the plxpr global indices of all the wires currently in the qreg.
        """
        return list(self.wire_map.keys())

    def global_index_to_local_index(self, global_index: int):
        """
        Convert a plxpr global index to the local index of this qreg.
        """
        return global_index - self.root_hash

    def local_index_to_global_index(self, index: int):
        """
        Convert a local index of this qreg to the plxpr global index.
        """
        return index + self.root_hash

    def __getitem__(self, index: int) -> AbstractQbit:
        """
        Get the newest ``AbstractQbit`` corresponding to a wire index.
        If the qubit value does not exist yet at this index, extract the fresh qubit.
        """
        global_index = self.local_index_to_global_index(index)
        if global_index in self.wire_map:
            return self.wire_map[global_index]
        return self.extract(index)

    def __setitem__(self, index: int, qubit: AbstractQbit):
        """
        Update the wire_map when a new qubit value for a wire index is produced,
        for example by gates.
        """
        global_index = self.local_index_to_global_index(index)
        self.wire_map[global_index] = qubit
