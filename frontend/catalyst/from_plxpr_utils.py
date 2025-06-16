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
    MeasurementPlane,
    compbasis_p,
    cond_p,
    counts_p,
    device_init_p,
    device_release_p,
    expval_p,
    for_p,
    gphase_p,
    measure_in_basis_p,
    measure_p,
    namedobs_p,
    probs_p,
    qalloc_p,
    qdealloc_p,
    qextract_p,
    qinsert_p,
    qinst_p,
    quantum_kernel_p,
    sample_p,
    set_basis_state_p,
    set_state_p,
    state_p,
    unitary_p,
    var_p,
    while_p,
)


class Qreg:
    """
    A manager that handles converting plxpr wire indices into catalyst jaxpr qreg.

    There are two ways a new qreg SSA value can be created in catalyst jaxpr:
    1. As the result of an allocation call;
    2. As a block argument for a subscope. For us, this means a qreg passed in as an argument on a
    subroutine, or a control flow's branch region.

    There are no quantum registers in plxpr. Instead, plxpr contains a list of gates and
    measurements applied to wires, with the wires being explicit numerical indices.
    To convert to catalyst jaxpr's qubit semantics, the central question is the following:

    * When an op on a given wire index is encountered in plxpr, what's the corresponding
    * SSA qubit value for that index in catalyst jaxpr?

    It is clear that at any point in the program, each wire (a qreg indexed into by a number)
    must have exactly one live qubit SSA value. In other words, when a gate acts on an index `i`,
    it should update the qubit value at index `i` on this register.

    This class keeps that map from the numerical indices to the SSA qubit values.
    """

    abstract_qreg_val: AbstractQreg
    wire_map: dict[int, AbstractQbit]  # Note: No dynamic wire indices for now in from_plxpr.

    def __init__(self, num_qubits, qreg_tracer=None):
        self.num_qubits = num_qubits

        # For qreg coming in as block arguments, the SSA qreg value would exist already.
        self.abstract_qreg_val = qreg_tracer
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

    def alloc(self):
        """Create the AbstractQreg from an alloc primitive"""
        self.abstract_qreg_val = qalloc_p.bind(self.num_qubits)

    def dealloc(self):
        """Create the dealloc primitive"""
        qdealloc_p.bind(self.abstract_qreg_val)

    def extract(self, wire: int) -> AbstractQbit:
        """Create the extract primitive that produces an AbstractQbit value."""

        # extract must be fresh
        assert wire not in self.wire_map
        extracted_qubit = qextract_p.bind(self.abstract_qreg_val, wire)
        self.wire_map[wire] = extracted_qubit
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

    def get_qubit_val_at_wire(self, wire: int) -> AbstractQbit:
        """Get the newest ``AbstractQbit`` corresponding to a wire index."""
        return self.wire_map[wire]

    def get_or_extract_val_at_wire(self, wire: int) -> AbstractQbit:
        """
        Get the newest ``AbstractQbit`` corresponding to a wire index.
        If the qubit value does not exist yet at this index, extract the fresh qubit.
        """
        if wire in self.wire_map:
            return self.get_qubit_val_at_wire(wire)
        return self.extract(wire)

    def update_qubit_val_at_wire(self, wire: int, qubit: AbstractQbit):
        """
        Update the wire_map when a new qubit value for a wire index is produced,
        for example by gates.
        """
        self.wire_map[wire] = qubit
