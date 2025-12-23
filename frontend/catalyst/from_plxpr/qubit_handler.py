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
1. Given a global wire index in plxpr, which logical catalyst register does this wire belong to?
2. Given a logical catalyst register, what is its current qreg SSA value, and what are its current
   qubit SSA values on its wires?
"""

from catalyst.jax_extras import DynamicJaxprTracer
from catalyst.jax_primitives import AbstractQbit, AbstractQreg, qextract_p, qinsert_p
from catalyst.utils.exceptions import CompileError

QREG_MIN_HASH = 1e12


class QubitIndexRecorder:
    """
    A manager that records what global plxpr wire index belongs to what logical register.

    This manager answers the first question.
    Since plxpr indices are global, there should be one instance of the class per
    PLxPRToQuantumJaxprInterpreter.

    This class promises the following:
    - `QubitIndexRecorder[i]` (i = global index in plxpr) returns the `QubitHandler` instance
       for the logical register that this wire belongs to.

    Users of this class are expected to:
    - Create an instance of this class per plxpr
    - Pass that instance to all `QubitHandler` instances created when tracing that plxpr
    """

    def __init__(self):
        self.map = {}

    def __getitem__(self, global_index):
        """
        Returns the `QubitHandler` instance that contains the requested global wire index.
        """
        assert global_index in self.map
        return self.map[global_index]

    def __setitem__(self, global_index, qreg_QubitHandler):
        """
        Set the corresponding quantum register `QubitHandler` for the provided global wire index.
        """
        self.map[global_index] = qreg_QubitHandler

    def contains(self, global_index):
        """
        Check if a global index is contained in any of the recorded qreg `QubitHandler`s.
        """
        return global_index in self.map


class QubitHandler:
    """
    A Qubit handler that manages converting plxpr wire indices into catalyst jaxpr qreg or qubits,
    depending on the context.

    Args:
        qubit_or_qreg_ref: An `AbstractQreg` value, or a list/tuple of `AbstractQbit` values.

    If `qubit_or_qreg_ref` is an `AbstractQreg`, this handler manages converting the qubits in qreg.

    However, if `qubit_or_qreg_ref` is a list/tuple of `AbstractQbit` values, this handler
    manages converting the qubits in the list/tuple. This is useful when the qubits
    are passed in as arguments to the function being converted. This feature is mainly
    useful for lowering decomposition rules that take in qubits as arguments.

    In qreg mode, this manager answers the second question, i.e. given a logical catalyst register,
    what is its current qreg SSA value, and what are its current qubit SSA values on its wires?

    This `QubitHandler` class promises the following in qreg mode:
    1. An instance of this class will always be tied to one root qreg value.
    2. At any moment during the from_plxpr conversion:
       - `QubitHandler[i]` (i=0,1,2....) returns the current catalyst qubit SSA value for the i-th
          index on the managed root register on that instance. If none exists, a new qubit will be
          extracted.
       - `QubitHandler.get()` returns the current catalyst qreg SSA value for the managed
          root register on that instance;

    To achieve the above, users of this class are expected to:
    1. Initialize an instance with the qreg SSA value from a new allocation or a new block argument;
    2. Whenever a new meta-op/op is bind-ed, update the current qreg/qubit SSA value with:
       - `QubitHandler.set(new_qreg_value)`
       - `QubitHandler[i] = new_qubit_value` (i=0,1,2....)
    """

    wire_map: dict[int | DynamicJaxprTracer, AbstractQbit]

    def __init__(
        self,
        qubit_or_qreg_ref: AbstractQreg | list[AbstractQbit] | tuple[AbstractQbit],
        recorder: QubitIndexRecorder,
        dynamically_alloced=False,
    ):

        # Qubit mode
        if isinstance(qubit_or_qreg_ref, (list, tuple)):
            self.abstract_qreg_val = None
            self.qubit_indices = qubit_or_qreg_ref
            # oddly enough we want to map the refs to themselves initially,
            # because when we interpret the plxpr it is done with the argument
            # types we want for the catalyst jaxpr (i.e. AbstractQbits),
            # so the original int[] invars are mapped to qubit tracers during the eval,
            # and each gate using the integer wires originally is interpreted with qubit
            # wires instead
            self.wire_map = dict(zip(self.qubit_indices, self.qubit_indices))
            return

        # Qreg mode
        else:
            # If this is an AbstractQreg, DynamicJaxprTracer, or None (for unit tests),
            self.qubit_indices = []
            self.abstract_qreg_val = qubit_or_qreg_ref
            self.recorder = recorder
            self.expired = False

            # A map from plxpr's *global* indices to the current catalyst SSA qubit values
            # for the wires on this qreg.
            self.wire_map = {}

            # Plxpr works with absolute indices, i.e. all unique, even across registers
            # However, catalyst qreg work with relative indices
            # i.e. extracting and inserting with every new catalyst qreg use indices 0,1,2,...
            # To distinguish between different registers, we require that each one has a hash.
            # For the static initial qreg, all the global indices in plxpr are 0, 1, 2... already
            # So for these non-dynamically allocated registers, the root hash is just zero, and
            # their local and global indices are the same.
            self.root_hash = (
                int(id(qubit_or_qreg_ref) + QREG_MIN_HASH) if dynamically_alloced else 0
            )

    def is_qubit_mode(self):
        """
        Returns True if the handler instance is in qubit mode, False if in qreg mode.
        """
        return self.abstract_qreg_val is None

    def get(self) -> AbstractQreg | list[AbstractQbit]:
        """
        Return the current AbstractQreg value or final AbstractQbit values
        depending on whichever is used to create the instance.
        """
        if self.is_qubit_mode():
            return [self.wire_map[idx] for idx in self.qubit_indices]
        else:
            return self.abstract_qreg_val

    def set(self, qreg: AbstractQreg):
        """
        Set the current AbstractQreg value.
        This is needed, for example, when exiting regions, like submodules and control flow.
        """

        if self.is_qubit_mode():
            # Devalidate the old qubit values if user wants to set a qreg
            self.wire_map = {}
            self.qubit_indices = []

        else:
            # The old qreg SSA value is no longer usable since a new one has appeared
            # Therefore all dangling qubits from the old one also all expire
            # These dangling qubit values will be dead, so there must be none.
            if len(self.wire_map) != 0:
                raise CompileError(
                    "Setting new qreg value, but the previous one still has dangling qubits."
                )
            self.abstract_qreg_val = qreg

    def __getitem__(self, local_index: int) -> AbstractQbit:
        """
        Get the newest ``AbstractQbit`` corresponding to a wire index, index = 0, 1, 2, ...
        If the qubit value does not exist yet at this index, extract the fresh qubit.
        """
        if self.is_qubit_mode():
            index = local_index
        else:
            index = self.local_index_to_global_index(local_index)

        if index in self.wire_map:
            return self.wire_map[index]

        return self.extract(local_index)

    def __setitem__(self, local_index: int, qubit: AbstractQbit):
        """
        Update the wire_map when a new qubit value for a wire index is produced,
        for example by gates.
        """
        if self.is_qubit_mode():
            index = local_index
        else:
            index = self.local_index_to_global_index(local_index)

        self.wire_map[index] = qubit

    def __iter__(self):
        """Iterate over wires map dictionary"""
        return iter(self.wire_map.items())

    def extract(self, index: int) -> AbstractQbit:
        """Create the extract primitive that produces an AbstractQbit value."""
        if self.is_qubit_mode():
            raise CompileError(
                f"Cannot extract a qubit at index {index} as there is no qreg."
                " Consider setting a qreg value first."
            )

        # Extract must be fresh
        global_index = self.local_index_to_global_index(index)
        assert global_index not in self.wire_map

        # record that this global wire index is on this register
        self.recorder[global_index] = self

        # extract and update current qubit value
        extracted_qubit = qextract_p.bind(self.abstract_qreg_val, index)
        self.wire_map[global_index] = extracted_qubit

        return extracted_qubit

    def insert(self, index: int, qubit: AbstractQbit):
        """
        Create the insert primitive.
        """
        if self.is_qubit_mode():
            raise CompileError(
                f"Cannot insert a qubit at index {index} as there is no qreg."
                " Consider setting a qreg value first."
            )

        global_index = self.local_index_to_global_index(index)
        self.abstract_qreg_val = qinsert_p.bind(self.abstract_qreg_val, index, qubit)
        self.wire_map.pop(global_index)

    def insert_all_dangling_qubits(self):
        """
        Insert all dangling qubits back into a qreg.

        This is necessary, for example, at the end of the qreg lifetime before deallocing,
        or when passing qregs into and out of scopes like control flow.
        """
        if self.is_qubit_mode():
            raise CompileError("Cannot insert qubits back into a qreg as there is no qreg.")

        for global_index, qubit in self.wire_map.items():
            if isinstance(global_index, DynamicJaxprTracer):
                # If tracer, user directly provides value during function their call
                # so no need to do anything
                idx = global_index
            else:
                idx = global_index - self.root_hash
            self.abstract_qreg_val = qinsert_p.bind(self.abstract_qreg_val, idx, qubit)

        self.wire_map.clear()

    def get_all_current_global_indices(self):
        """
        Return a list of the plxpr global indices of all the wires currently in the qreg.
        """
        return list(self.wire_map.keys())

    def global_index_to_local_index(self, global_index: int | DynamicJaxprTracer):
        """
        Convert a plxpr global index to the local index of this qreg.
        """
        if isinstance(global_index, DynamicJaxprTracer):
            # If tracer, user directly provides value during function their call
            # so no need to do anything
            return global_index
        return global_index - self.root_hash

    def local_index_to_global_index(self, local_index: int | DynamicJaxprTracer):
        """
        Convert a local index of this qreg to the plxpr global index.
        """
        if isinstance(local_index, DynamicJaxprTracer):
            # If tracer, user directly provides value during function their call
            # so no need to do anything
            return local_index
        return local_index + self.root_hash

    def insert_dynamic_qubits(self, wires):
        """
        When wire label is dynamic, such gates will need to interrupt analysis like cancel inverses:
           qml.X(wires=0)
           qml.Hadamard(wires=w)
           qml.X(wires=0)
        In the above, because we don't know whether `w` is `0` or not until runtime, we must not
        cancel the inverse PauliX gates on `0`.

        Thus in the IR def use chain we need to interrupt the first X gate's %out_qubit to be used
        as the second X gate's qubit operand directly.
        To do this, for the dynamic wire gate we insert back to the register.
        """
        if self.is_qubit_mode():
            raise CompileError("Cannot insert dynamic qubits back into a qreg as there is no qreg.")

        # Cancel-inverses style passes only work on gates with same number of qubits
        same_number_of_wires = len(wires) == len(self.wire_map)

        all_static_requested = all(isinstance(wire, int) for wire in wires)
        all_static_cached = all(isinstance(wire, int) for wire in self.wire_map.keys())
        all_static = all_static_requested and all_static_cached

        if all_static:
            # no need to insert back anything if nothing is dynamic
            return

        all_dynamic = False
        keep_cache = False
        if same_number_of_wires:
            # Notice that we can keep using the current qubit value in the wire_map (if one exists
            # there) even for dynamic case if they are the same dynamic wire, i.e.
            #   qml.gate(wires=[w0,w1])
            #   qml.gate(wires=[w0,w1])
            # these cases do not need to insert back

            wires_all_in_cache = all(wire in self.wire_map.keys() for wire in wires)
            all_dynamic = all(not isinstance(wire, int) for wire in wires)
            keep_cache = wires_all_in_cache and all_dynamic

        if not keep_cache:
            self.insert_all_dangling_qubits()


def is_dynamically_allocated_wire(wire):
    """
    Return whether a given global wire index comes from a dynamical allocation.
    """
    return isinstance(wire, int) and wire > QREG_MIN_HASH


def get_in_qubit_values(
    wires, qubit_index_recorder: QubitIndexRecorder, fallback_qreg: QubitHandler
):
    """
    Return the current SSA qreg and qubit values corresponding to the plxpr global indices `wires`.

    Args:
        wires (Iterable): the global plxpr indices to be queried.
        qubit_index_recorder (QubitIndexRecorder): the recorder of the current interpreter.
        fallback_qreg (QubitHandler): the qreg to extract qubits from if the requested plxpr wire
        indices have not been recorded on the recorder.

    Returns:
        in_qregs (List): the i-th entry of this result is the current qreg SSA value for the
        register containing the i-th wire in the argument.
        in_qubits (List): the i-th entry of this result is the current qubit SSA value for the
        i-th wire in the argument.
    """
    in_qregs = []
    in_qubits = []

    for w in wires:
        if not qubit_index_recorder.contains(w):
            # First time the global wire index w is encountered
            # Need to extract from fallback qreg
            in_qubits.append(fallback_qreg[fallback_qreg.global_index_to_local_index(w)])
            in_qregs.append(fallback_qreg)

        else:
            in_qreg = qubit_index_recorder[w]
            in_qregs.append(in_qreg)
            in_qubits.append(in_qreg[in_qreg.global_index_to_local_index(w)])

    return in_qregs, in_qubits


def _get_dynamically_allocated_qregs(plxpr_invals, qubit_index_recorder, init_qreg):
    """
    Get the potential dynamically allocated register values that are visible to a jaxpr.

    Note that dynamically allocated wires have their qreg tracer's id as the global wire index
    so the sub jaxpr takes that id in as a "const" (if it is one; as opposed to tracers),
    since it is closure from the target wire of gates/measurements/...

    We need to remove that const, so we also let this util return these global indices.
    """
    dynalloced_qregs = []
    dynalloced_wire_global_indices = []
    for inval in plxpr_invals:
        if not type(inval) in [int, DynamicJaxprTracer]:
            # don't care about invals that won't be wire indices
            continue
        if qubit_index_recorder.contains(inval) and qubit_index_recorder[inval] is not init_qreg:
            dyn_qreg = qubit_index_recorder[inval]
            dyn_qreg.insert_all_dangling_qubits()
            dynalloced_qregs.append(dyn_qreg)
            if isinstance(inval, int):
                dynalloced_wire_global_indices.append(inval)

    return dynalloced_qregs, dynalloced_wire_global_indices
