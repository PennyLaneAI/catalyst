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
  * 1. Only observables PauliX, PauliY, PauliZ, Hadamard and Identity are currently supported when
    using this transform (but these are also the only observables currently supported in the
    Quantum dialect as NamedObservable).
  * 2. Unlike the current tape-based implementation of the transform, it doesn't allow for
    diagonalization of a subset of observables.
  * 3. Unlike the current tape-based implementation of the transform, conversion to measurements
    based on eigvals and wires (rather than the PauliZ observable) is not currently supported.
  * 4. Unlike the tape-based implementation, this pass will NOT raise an error if given a circuit
    that is invalid because it contains non-commuting measurements. It should be assumed that
    this transform results in incorrect outputs unless split_non_commuting is applied to break
    non-commuting measurements into separate tapes.
    - We need to check if a circuit contains non-commuting measurements or not

Tasks
----------------
  * Limitations 2 & 3 requires the feature allowing us to pass options to the xdsl pass.
  Example [here](https://github.com/PennyLaneAI/catalyst/blob/195d8fa2f307dae8362dad123d56a83c4ade9779/frontend/test/pytest/python_interface/pass_api/test_transform_interpreter.py#L168)_
  * Limitations 4 requires a step to detect if the all measurements in the circuit commute with each other or not.

"""

from dataclasses import dataclass

from pennylane.ops import Hadamard, PauliX, PauliY
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import arith, builtin
from xdsl.rewriter import InsertPoint

from catalyst.python_interface.dialects.quantum import (
    CustomOp,
    GlobalPhaseOp,
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


class DiagonalizeFinalMeasurementsPattern(
    pattern_rewriter.RewritePattern
):  # pylint: disable=too-few-public-methods
    """RewritePattern for diagonalizing final measurements."""

    def __init__(self, supported_base_obs, to_eigvals):
        self.supported_base_obs = supported_base_obs
        self.to_eigvals = to_eigvals

    @pattern_rewriter.op_type_rewrite_pattern
    def match_and_rewrite(
        self, observable: NamedObsOp, rewriter: pattern_rewriter.PatternRewriter, /
    ):
        """Replace non-diagonalized observables with their diagonalizing gates and PauliZ."""

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
            num_observables = len(
                [use for use in uses_to_change if isinstance(use.operation, NamedObsOp)]
            )

            if num_observables > 1:
                raise RuntimeError(
                    "Each wire can only have one set of diagonalizing gates applied, but the "
                    "circuit contains multiple observables with the same wire."
                )

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
        self.supported_base_obs = (
            options["supported-base-obs"]
            if "supported-base-obs" in options and options["supported-base-obs"] is not None
            else _default_supported_obs
        )
        if "to-eigvals" in options and options["to-eigvals"] is True:
            raise ValueError("to_eigvals = True is not supported")
        self.to_eigvals = False

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """Apply the diagonalize final measurements pass."""
        pattern_rewriter.PatternRewriteWalker(
            DiagonalizeFinalMeasurementsPattern(self.supported_base_obs, self.to_eigvals)
        ).rewrite_module(op)


diagonalize_final_measurements_pass = compiler_transform(DiagonalizeFinalMeasurementsPass)
