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
from collections.abc import Collection

from pennylane.exceptions import CompileError
from pennylane.ops import Hadamard, PauliX, PauliY
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin, func
from xdsl.rewriter import InsertPoint
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination

from catalyst.python_interface.dialects.quantum import (
    ComputationalBasisOp,
    CustomOp,
    GlobalPhaseOp,
    HermitianOp,
    MultiRZOp,
    NamedObservable,
    NamedObservableAttr,
    NamedObsOp,
    QubitUnitaryOp,
)
from catalyst.python_interface.pass_api import compiler_transform

_default_supported_obs = {"PauliZ", "Identity"}


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


class NonCommutingObservableValidator:
    """
    Validates if all quantum observables of an operation commute.
    """

    _error_msg = (
        "Observables are not qubit-wise commuting. Please apply the "
        "`split-non-commuting` pass first."
    )

    def __init__(self, op, supported_base_obs: Collection[str]):
        self.op = op
        self.supported_base_obs = set(supported_base_obs)

        # Core State
        self.visited_qubits = set()
        self.overlapped_qubits = set()
        self.obs_on_qubits = defaultdict(set)
        self.visited_qreg = False
        self.is_qwc_compatible = True

        # 1. Single pass to gather all required IR metadata
        self._analyze_module()

        if self.is_qwc_compatible:
            self._run_qwc_validation()
        else:
            self._run_non_overlapping_validation()

    def _analyze_module(self):
        """Single walk through the IR to populate qubit and observable maps."""
        for op_ in self.op.walk():
            if isinstance(op_, NamedObsOp):
                self._register_qubits([op_.qubit], op_.type.data.value)

            elif isinstance(op_, HermitianOp):
                self.is_qwc_compatible = False
                self._register_qubits(op_.qubits, "Hermitian")

            elif isinstance(op_, ComputationalBasisOp):
                if op_.qreg:
                    self.visited_qreg = True
                if op_.qubits:
                    self._register_qubits(op_.qubits, "PauliZ")

    def _register_qubits(self, qubits, obs_type):
        """Updates tracking for SSA qubit values and their associated observables."""
        for q in qubits:
            if q in self.visited_qubits:
                self.overlapped_qubits.add(q)
            self.visited_qubits.add(q)
            self.obs_on_qubits[q].add(obs_type)

    def _run_qwc_validation(self):
        """Validates Qubit-Wise Commutativity logic."""
        success = True
        if self.visited_qreg:
            # If a register is used, all ops must be Z-basis or Identity
            for obs_set in self.obs_on_qubits.values():
                if not obs_set.issubset({"PauliZ", "Identity"}):
                    success = False
        else:
            # Check overlapping qubits for conflicts (more than 1 non-Identity)
            for q in self.overlapped_qubits:
                obs = self.obs_on_qubits[q]
                if len(obs - {"Identity"}) > 1:
                    success = False
        if not success:
            # Fallback to stricter check if QWC logic fails
            self._run_non_overlapping_validation()

    def _run_non_overlapping_validation(self):
        """Strictest check: no two observables can share a qubit."""
        if self.overlapped_qubits or (self.visited_qreg and self.obs_on_qubits):
            raise CompileError(self._error_msg)


class DiagonalizeFinalMeasurementsPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for diagonalizing final measurements."""

    def __init__(self, supported_base_obs: set[str]):
        """Initializes the RewritePattern."""
        self.supported_base_obs = supported_base_obs

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, quantum_funcOp: func.FuncOp, rewriter: pattern_rewriter.PatternRewriter, /
    ):
        """Replace non-diagonalized observables with their diagonalizing gates and supported
        base observables."""

        for op in quantum_funcOp.body.walk():
            if isinstance(op, NamedObsOp) and _diagonalize(op, self.supported_base_obs):

                diagonalizing_gates = _gate_map[op.type.data]
                params = _params_map[op.type.data]

                qubit = op.qubit

                insert_point = InsertPoint.before(op)

                # For commuting observables, we only need to diagonalize the first observable
                # encountered in the IR for each qubit. With the current disign of observable
                # operations, same observables acting at the same qubit are not reused in the
                # measurement operations. Instead, the qubit ssa value is reused. Consequently,
                # we could only diagonalize the first encountered observable for a qubit and
                # then update the sequent observables act on the same qubit. The following patch
                # collects the sequent observables act on the same qubit for the update later.
                commute_obs = [
                    use.operation
                    for use in op.qubit.uses
                    if isinstance(use.operation, (NamedObsOp)) and use.operation is not op
                ]

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

                # we need to replace the initial qubit use everywhere EXCEPT the use that is now the
                # input to the first diagonalizing gate. It's not enough to only change the
                # NamedObsOp, because the qubit might be inserted/deallocated later
                uses_to_change = [
                    use
                    for use in op.qubit.uses
                    if not isinstance(
                        use.operation, (CustomOp, GlobalPhaseOp, MultiRZOp, QubitUnitaryOp)
                    )
                ]

                # pylint: disable = cell-var-from-loop
                op.qubit.replace_uses_with_if(qubit, lambda use: use in uses_to_change)
                for use in uses_to_change:
                    rewriter.notify_op_modified(use.operation)

                # then we also update the observable to be in the Z basis. Since this is done with
                # the rewriter, we don't need to call `rewriter.notify_modified(observable)`
                # regarding this
                diag_obs = NamedObsOp(
                    qubit=qubit, obs_type=NamedObservableAttr(NamedObservable("PauliZ"))
                )
                rewriter.replace_op(op, diag_obs)

                # update the sequent observables act on the same qubit to be in Z basis, if
                # the observable is same as the current op.
                for ob in commute_obs:
                    if ob.type.data.value == op.type.data.value:
                        diag_obs = NamedObsOp(
                            qubit=qubit, obs_type=NamedObservableAttr(NamedObservable("PauliZ"))
                        )
                        rewriter.replace_op(ob, diag_obs)


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

        CommonSubexpressionElimination().apply(_ctx, op)

        for op_ in op.walk():
            if isinstance(op_, func.FuncOp) and "quantum.node" in op_.attributes:
                # Validate if each circuit in the module is commuting and an error
                # will raise if not.
                NonCommutingObservableValidator(op_, self.supported_base_obs)

                rewriter = pattern_rewriter.PatternRewriter(op_)

                DiagonalizeFinalMeasurementsPattern(self.supported_base_obs).match_and_rewrite(
                    op_, rewriter
                )


diagonalize_final_measurements_pass = compiler_transform(DiagonalizeFinalMeasurementsPass)
