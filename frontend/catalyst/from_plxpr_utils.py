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
    Note: No dynamic wire indices for now in from_plxpr.
    """

    abstract_qreg_val: AbstractQreg
    wire_map: dict[int, AbstractQbit]

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.abstract_qreg_val = None

        # plxpr always uses wire index (numbers)
        # This map would record the wire index to the corrsponding "newest" abstract qubit value.
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

    def insert_all_dangling_qubits(self):
        """
        Insert all dangling qubits back into a qreg.

        This is necessary, for example, at the end of the qreg lifetime before deallocing,
        or when passing qregs into and out of scopes like control flow.
        """
        for index, qubit in self.wire_map.items():
            self.insert(index, qubit)

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
