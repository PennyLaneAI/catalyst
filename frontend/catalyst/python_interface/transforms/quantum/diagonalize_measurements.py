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
"""This file contains the implementation of the diagonalize_final_measurements transform,
written using xDSL.


Known Limitations
-----------------
  * Only observables PauliX, PauliY, PauliZ, Hadamard and Identity are currently supported when
    using this transform (but these are also the only observables currently supported in the
    Quantum dialect as NamedObservable).
  * Unlike the current tape-based implementation of the transform, conversion to measurements
    based on eigvals and wires (rather than the PauliZ observable) is not currently supported.
    If `eigvals=True` is passed to the current pass, an error will be raised.
"""

from collections import defaultdict

from pennylane.ops import Hadamard, PauliX, PauliY
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin
from xdsl.rewriter import InsertPoint

from catalyst.python_interface.dialects.quantum import (
    ComputationalBasisOp,
    CustomOp,
    GlobalPhaseOp,
    HamiltonianOp,
    HermitianOp,
    MultiRZOp,
    NamedObservable,
    NamedObservableAttr,
    NamedObsOp,
    QubitUnitaryOp,
    TensorOp,
    TerminalMeasurementOp,
)
from catalyst.python_interface.pass_api import compiler_transform


def _generate_mapping():
    _gate_map = {}
    _params_map = {}

    for op in PauliX(0), PauliY(0), Hadamard(0):
        diagonalizing_gates = op.diagonalizing_gates()

        _gate_map[op.name] = [gate.name for gate in diagonalizing_gates]
        _params_map[op.name] = [gate.data for gate in diagonalizing_gates]

    return _gate_map, _params_map


_gate_map, _params_map = _generate_mapping()


def _diagonalize(obs: NamedObsOp, supported_base_obs) -> bool:
    """Whether to diagonalize a given observable."""
    if obs.type.data in supported_base_obs:
        return False
    if obs.type.data in _gate_map:
        return True
    raise NotImplementedError(
        f"Observable {obs.type.data} is not supported for diagonalization"
    )  # pragma: no cover


from collections import defaultdict


class NonCommutingObservableValidator:
    """
    Validates if all quantum observables in a builtin.ModuleOp object act on
    disjoint qubit sets.
    """

    def __init__(self, ops: builtin.ModuleOp):
        self._ops = ops
        is_commute_rule = self._check_obs_commuting()
        if not is_commute_rule:
            self._check_obs_wire_overlapping()

    @property
    def _err_overlapping_msg(self):
        """Error message to be raised for"""
        return (
            "Only observables that act on non-overlapping qubits can be diagonalized. "
            "Please apply the `split-non-commuting` pass first."
        )

    @property
    def _err_qwc_msg(self):
        """Error message to be raised for"""
        return (
            "Only observables are not commuting can be diagonalized. "
            "Please apply the `split-non-commuting` pass first."
        )

    def _walk_ssa_to_pauliword_op(self, original_ssa, pauli_obs, ssa, final_obs):
        for use in ssa.uses:
            current_op = use.operation
            if isinstance(current_op, NamedObsOp) and pauli_obs is not current_op.type.data.value:
                pauli_obs = current_op.type.data.value
                self._walk_ssa_to_pauliword_op(original_ssa, pauli_obs, current_op.obs, final_obs)
            elif isinstance(current_op, ComputationalBasisOp):
                pauli_obs = "PauliZ"
                self._walk_ssa_to_pauliword_op(original_ssa, pauli_obs, current_op.obs, final_obs)
            elif isinstance(current_op, TerminalMeasurementOp):
                final_obs[ssa][original_ssa] = pauli_obs
            elif isinstance(current_op, TensorOp):
                self._walk_ssa_to_pauliword_op(original_ssa, pauli_obs, current_op.obs, final_obs)

    def _is_qwc_required(
        self,
    ):
        # Follow the diagonalize_qwc_pauli_words implementation, it only detects commutative
        # Pauli words. If there is a non-Pauli word obserable in the module, we do the qubit-
        # overlapping check instead. The following block also check if there is qubit-overlapping
        # in the observables
        is_commute_rule = True
        visited_qubits = set()
        overlapped_qubits = set()
        visited_qreg = False
        for op in self._ops.walk():
            if isinstance(op, NamedObsOp):
                if op.qubit not in visited_qubits:
                    visited_qubits.add(op.qubit)
                else:
                    overlapped_qubits.add(op.qubit)
            elif isinstance(op, HermitianOp):
                is_commute_rule = False
                break
            elif isinstance(op, HamiltonianOp) and len(op.terms) > 1:
                is_commute_rule = False
                break
            elif isinstance(op, ComputationalBasisOp):
                if op.qreg is not None:
                    visited_qreg = True
                if op.qubits is not None:
                    for qubit in op.qubits:
                        if qubit not in visited_qubits:
                            visited_qubits.add(qubit)
                        else:
                            overlapped_qubits.add(qubit)

        if is_commute_rule:
            if not visited_qreg and len(overlapped_qubits) == 0:
                is_commute_rule = False
            if visited_qreg and len(visited_qubits) == 0:
                is_commute_rule = False

        return is_commute_rule, overlapped_qubits, visited_qreg, visited_qubits

    def _collect_obs_with_overlapping_qubit(self, overlapped_qubits):
        final_obs_dict = defaultdict(defaultdict)
        for qubit in overlapped_qubits:
            self._walk_ssa_to_pauliword_op(qubit, None, qubit, final_obs_dict)
        return final_obs_dict

    def _validate_qwc_obs_with_overlapping_qubit(self, obs_dict):
        for obs0 in obs_dict:
            for obs1 in obs_dict:
                if obs0 is not obs1:
                    for qubit in obs_dict[obs0]:
                        pauli0 = obs_dict[obs0][qubit]
                        pauli1 = obs_dict[obs1].get(qubit, "Identity")
                        if "Identity" not in (pauli0, pauli1) and pauli0 != pauli1:
                            raise RuntimeError(self._err_qwc_msg)

    def _validate_qwc_obs_with_qreg(self, obs_dict):
        for obs in obs_dict.values():
            for obs_name in obs.values():
                if obs_name not in ("PauliZ", "Identity"):
                    raise RuntimeError(self._err_qwc_msg)

    def _check_obs_commuting(self):
        # Get measurements with obs sharing wires
        is_commute_rule, overlapped_qubits, visited_qreg, visited_qubits = self._is_qwc_required()
        if is_commute_rule:
            obs_dict = self._collect_obs_with_overlapping_qubit(overlapped_qubits)
            self._validate_qwc_obs_with_overlapping_qubit(obs_dict)

            if visited_qreg:
                obs_dict = self._collect_obs_with_overlapping_qubit(visited_qubits)
                self._validate_qwc_obs_with_qreg(obs_dict)

        return is_commute_rule

    def _check_obs_wire_overlapping(self):
        """Dispatches the observable to the correct qubit/qreg tracking logic."""
        _visited_qubits = set()
        _visited_qreg = False

        for op in self._ops.walk():
            if isinstance(op, NamedObsOp):
                self._update_visited_qubits([op.qubit], _visited_qubits, _visited_qreg)
            elif isinstance(op, HermitianOp):
                self._update_visited_qubits(op.qubits, _visited_qubits, _visited_qreg)
            elif isinstance(op, ComputationalBasisOp):
                if op.qreg is not None:
                    self._update_visited_qreg(_visited_qubits, _visited_qreg)
                if op.qubits is not None:
                    self._update_visited_qubits(op.qubits, _visited_qubits, _visited_qreg)

    def _update_visited_qubits(self, qubits, _visited_qubits, _visited_qreg):
        """Checks if the specific qubits have already been acted upon.

        Args:
            qubits: An iterable of qubit SSAValue.
        """
        for qubit in qubits:
            if _visited_qreg or qubit in _visited_qubits:
                raise RuntimeError(f"{self._err_overlapping_msg}")
            _visited_qubits.add(qubit)

    def _update_visited_qreg(self, _visited_qubits, _visited_qreg):
        """Checks if the qreg can be visited. Fails if any individual qubits or
        another register have already been visited."""

        if _visited_qreg or _visited_qubits:
            raise RuntimeError(f"{self._err_overlapping_msg}")
        _visited_qreg = True


class DiagonalizeFinalMeasurementsPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for diagonalizing final measurements."""

    def __init__(self, supported_base_obs: set[str]):
        """Initializes the RewritePattern."""
        self.supported_base_obs = supported_base_obs

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, observable: NamedObsOp, rewriter: pattern_rewriter.PatternRewriter, /
    ):
        """Replace non-diagonalized observables with their diagonalizing gates and supported
        base observables."""

        if _diagonalize(observable, self.supported_base_obs):

            diagonalizing_gates = _gate_map[observable.type.data]
            params = _params_map[observable.type.data]

            qubit = observable.qubit

            insert_point = InsertPoint.before(observable)

            for name, op_data in zip(diagonalizing_gates, params):
                if op_data:
                    param_ssa_values = []
                    for param in op_data:
                        paramOp = arith.ConstantOp(
                            builtin.FloatAttr(data=param, type=builtin.Float64Type())
                        )
                        rewriter.insert_op(paramOp, insert_point)
                        param_ssa_values.append(paramOp.results[0])

                    gate = CustomOp(in_qubits=qubit, gate_name=name, params=param_ssa_values)
                else:
                    gate = CustomOp(in_qubits=qubit, gate_name=name)

                rewriter.insert_op(gate, insert_point)

                qubit = gate.out_qubits[0]

            # we need to replace the initial qubit use everwhere EXCEPT the use that is now the
            # input to the first diagonalizing gate. Its not enough to only change the NamedObsOp,
            # because the qubit might be inserted/deallocated later
            uses_to_change = [
                use
                for use in observable.qubit.uses
                if not isinstance(
                    use.operation, (CustomOp, GlobalPhaseOp, MultiRZOp, QubitUnitaryOp)
                )
            ]

            observable.qubit.replace_by_if(qubit, lambda use: use in uses_to_change)
            for use in uses_to_change:
                rewriter.notify_op_modified(use.operation)

            # then we also update the observable to be in the Z basis. Since this is done with the
            # rewriter, we don't need to call `rewriter.notify_modified(observable)` regarding this
            diag_obs = NamedObsOp(
                qubit=qubit, obs_type=NamedObservableAttr(NamedObservable("PauliZ"))
            )
            rewriter.replace_op(observable, diag_obs)


class DiagonalizeFinalMeasurementsPass(passes.ModulePass):
    """Pass for diagonalizing final measurements."""

    name = "diagonalize-final-measurements"

    def __init__(self, **options):
        """Initializes the class with supported base observable names and .

        Args:
            **options: Arbitrary keyword arguments.
                supported_base_obs (tuple[str], optional): The observable bases
                    to support. Must be a subset of the allowed base observables
                    (e.g., 'PauliX', 'PauliY', 'PauliZ', 'Hadamard', 'Identity').
                    Defaults to `_default_supported_obs`.
                to_eigvals (bool, optional): Whether to convert to eigenvalues.
                    Currently, only `False` is supported. Defaults to `False`.

        Raises:
            ValueError: If `supported_base_obs` contains observables not found in
                `_obs_allowed_diagonalization`.
            ValueError: If `to_eigvals` is set to `True`.
        """
        _default_supported_obs = {"PauliZ", "Identity"}
        _obs_allowed_diagonalization = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Identity"}

        self.supported_base_obs = options.get("supported_base_obs", tuple(_default_supported_obs))

        if isinstance(self.supported_base_obs, tuple) and set(self.supported_base_obs).issubset(
            _obs_allowed_diagonalization
        ):
            self.supported_base_obs = _default_supported_obs | set(self.supported_base_obs)
        else:
            msg = (
                "Supported base observables must be a subset of (PauliX, PauliY, PauliZ, Hadamard, "
                "and Identity) passed as a tuple[str], but received "
                f"{self.supported_base_obs}"
            )
            raise ValueError(msg)
        if options.get("to_eigvals", False) is not False:
            raise ValueError("Only to_eigvals = False is supported.")

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the diagonalize final measurements pass."""
        # Validate if the circuit in the module is commuting and an error
        # will raise if not.
        NonCommutingObservableValidator(op)

        pattern_rewriter.PatternRewriteWalker(
            DiagonalizeFinalMeasurementsPattern(self.supported_base_obs)
        ).rewrite_module(op)


diagonalize_final_measurements_pass = compiler_transform(DiagonalizeFinalMeasurementsPass)
