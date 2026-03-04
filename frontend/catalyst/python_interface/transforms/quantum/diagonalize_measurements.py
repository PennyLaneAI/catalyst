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

from pennylane.ops import Hadamard, PauliX, PauliY
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin
from xdsl.rewriter import InsertPoint

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
_obs_allowed_diagonalization = {"PauliX", "PauliY", "PauliZ", "Hadamard", "Identity"}


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


class CheckSplitNonCommutingPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for diagonalizing final measurements."""

    def __init__(self):
        self._visited_obs_qubits = set()
        self._visited_obs_qreg = False

    def _update_visited_qubits(self, qubits):
        for qubit in qubits:
            if self._visited_obs_qreg or qubit in self._visited_obs_qubits:
                raise RuntimeError("cannot diagonalize circuit with non-commuting observables")
            self._visited_obs_qubits.add(qubit)

    def _update_visited_qreg(self):
        if self._visited_obs_qreg or self._visited_obs_qubits:
            raise RuntimeError("cannot diagonalize circuit with non-commuting observables")
        self._visited_obs_qreg = True

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        observable: NamedObsOp | ComputationalBasisOp | HermitianOp,
        rewriter: pattern_rewriter.PatternRewriter,
        /,
    ):
        """Check if the circuit is commuting or not."""
        if isinstance(observable, NamedObsOp):
            self._update_visited_qubits([observable.operands[0]])

        elif isinstance(observable, HermitianOp):
            self._update_visited_qubits(observable.qubits)

        elif isinstance(observable, ComputationalBasisOp):
            if observable.qreg is not None:
                self._update_visited_qreg()

            if observable.qubits is not None:
                self._update_visited_qubits(observable.qubits)


class DiagonalizeFinalMeasurementsPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for diagonalizing final measurements."""

    def __init__(self, supported_base_obs: set[str], to_eigvals: bool = False):
        """Initializes the RewritePattern."""
        self.supported_base_obs = supported_base_obs
        self.to_eigvals = to_eigvals

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
                supported_base_obs (str or tuple[str], optional): The observable bases
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

        self.supported_base_obs = options.get("supported_base_obs", _default_supported_obs)

        if (
            isinstance(self.supported_base_obs, str)
            and self.supported_base_obs in _obs_allowed_diagonalization
        ):
            self.supported_base_obs = _default_supported_obs | set(
                [
                    self.supported_base_obs,
                ]
            )
        if isinstance(self.supported_base_obs, tuple) and set(self.supported_base_obs).issubset(
            _obs_allowed_diagonalization
        ):
            self.supported_base_obs = _default_supported_obs | set(self.supported_base_obs)

        if not set(self.supported_base_obs).issubset(_obs_allowed_diagonalization):
            msg = (
                "Supported base observables must be a subset of (PauliX, PauliY, PauliZ, Hadamard, "
                "and Identity) passed as a tuple[str] or str, but received "
                f"{self.supported_base_obs}"
            )
            raise ValueError(msg)
        self.to_eigvals = options.get("to_eigvals", False)
        if self.to_eigvals is not False:
            raise ValueError("Only to_eigvals = False is supported.")

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the diagonalize final measurements pass."""
        pattern_rewriter.PatternRewriteWalker(
            CheckSplitNonCommutingPattern(), apply_recursively=False
        ).rewrite_module(op)

        pattern_rewriter.PatternRewriteWalker(
            DiagonalizeFinalMeasurementsPattern(self.supported_base_obs, self.to_eigvals)
        ).rewrite_module(op)


diagonalize_final_measurements_pass = compiler_transform(DiagonalizeFinalMeasurementsPass)
