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
"""Cancel inverses prototype using the new pass API."""
# pylint: disable=unused-argument

from xdsl.ir import Operation

from catalyst.python_interface.dialects import quantum

from .compiler_transform import CompilationPass, compiler_transform


class CancelInverses(CompilationPass):
    """Reimplementation of cancel_inverses using new API."""

    name = "cancel-inverses-2"

    self_inverses = [
        "Identity",
        "Hadamard",
        "PauliX",
        "PauliY",
        "PauliZ",
        "CNOT",
        "CZ",
        "CY",
        "CH",
        "SWAP",
        "Toffoli",
        "CCZ",
    ]

    @staticmethod
    def can_cancel(op: quantum.CustomOp, next_op: Operation) -> bool:
        """Check if ops can be cancelled."""
        if isinstance(next_op, quantum.CustomOp):
            if op.gate_name.data == next_op.gate_name.data:
                if (
                    op.out_qubits == next_op.in_qubits
                    and op.out_ctrl_qubits == next_op.in_ctrl_qubits
                    and op.in_ctrl_values == next_op.in_ctrl_values
                ):
                    return True

        return False

    def action(self, op: quantum.CustomOp, rewriter):
        """Rewrite rule for CustomOp."""
        while isinstance(op, quantum.CustomOp) and op.gate_name.data in self.self_inverses:
            next_user = None
            for use in op.results[0].uses:
                user = use.operation
                if self.can_cancel(op, user):
                    next_user: quantum.CustomOp = user
                    break

            if next_user is None:
                break

            for q1, q2 in zip(op.in_qubits, next_user.out_qubits, strict=True):
                rewriter.replace_all_uses_with(q2, q1)
            for cq1, cq2 in zip(op.in_ctrl_qubits, next_user.out_ctrl_qubits, strict=True):
                rewriter.replace_all_uses_with(cq2, cq1)
            rewriter.erase_op(next_user)
            rewriter.erase_op(op)
            op = op.in_qubits[0].owner


# We can register more rewrite rules as needed. Here are some
# dummy rewrite rules to illustrate:
@CancelInverses.add_action
def rewrite_insert_op(self, op: quantum.InsertOp, rewriter):
    """Rewrite rule for InsertOp."""
    return


@CancelInverses.add_action
def rewrite_extract_op(self, op: quantum.ExtractOp, rewriter):
    """Rewrite rule for ExtractOp."""
    return


@CancelInverses.add_action
def rewrite_mid_measure_op(self, op: quantum.MeasureOp, rewriter):
    """Rewrite rule for MeasureOp."""
    return


# Unions of operation types can also be used
@CancelInverses.add_action
def rewrite_observable_op(
    self,
    op: (
        quantum.HermitianOp
        | quantum.NamedObsOp
        | quantum.ComputationalBasisOp
        | quantum.HamiltonianOp
        | quantum.TensorOp
    ),
    rewriter,
):
    """Rewrite rule for observable ops."""
    return


cancel_inverses_2 = compiler_transform(CancelInverses)
