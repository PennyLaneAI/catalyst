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

"""This module contains a Pattern for wrapping the QNode in a classical function that
calls the QNode, so that post-processing can be injected after the QNode call.
"""

from pennylane.exceptions import CompileError
from xdsl.ir import Operation
from xdsl.dialects import builtin, func
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint


def get_call_op(qnode: func.FuncOp):
    """Get the CallOp in another module function that calls this quantum_node. Postprocessing
    will be called to act on the output of that CallOp"""
    # ToDo: improve docstring so its clear what this is for. Probably add an example.

    module = _get_parent_module(qnode)
    qnode_name = qnode.sym_name.data
    all_call_ops = [op for op in module.body.walk() if isinstance(op, func.CallOp)]
    qnode_call_op = [op for op in all_call_ops if qnode_name in op.callee.string_value()]
    if not len(qnode_call_op) == 1:
        raise CompileError(
            f"Expected only one call_op for {qnode_name}, but received {qnode_call_op}"
        )

    return qnode_call_op[0]


class OutlineQNodePattern(RewritePattern):
    """RewritePattern putting a quantum.node into a classical function that
    calls the quantum.node. Postprocessing can then be inserted after the call
    by Patterns that add post-processing."""

    # ToDo: add example, and include pass_str arg in the docstring

    def __init__(self, pass_str: str):
        super().__init__()
        self._pass_str = pass_str

    def match_and_rewrite(self, func_op: func.FuncOp, rewriter: PatternRewriter):
        """Transform a quantum node (qnode) to separate it into a quantum node
        and an outer function that calls the quantum node. Subsequent patterns can then apply
        postprocessing to the results of the call in the outer function when changes are made
        to the quantum node.

        The name of the quantum node is changed to original_name.pass_str, and the new outer
        postprocessing function is given the original name.
        """

        if "quantum.node" not in func_op.attributes:
            return

        # get names of the postprocessing and quantum_node functions
        outer_fn_name = func_op.sym_name.data
        qnode_name = outer_fn_name + f".{self._pass_str}"

        # update quantum_node name to include pass string
        func_op.sym_name = builtin.StringAttr(qnode_name)
        rewriter.notify_op_modified(func_op)

        # create the outer (postprocessing) fn with the original node name
        outer_fn = func.FuncOp(
            name=outer_fn_name, function_type=func_op.function_type, visibility="public"
        )
        rewriter.insert_op(outer_fn, InsertPoint.before(func_op))

        # call the renamed quantum_node inside the new outer FuncOp
        call_args = outer_fn.body.block.args
        result_types = outer_fn.function_type.outputs.data
        call_op = func.CallOp(qnode_name, call_args, result_types)
        outer_fn.body.block.add_op(call_op)

        # add a ReturnOp in the new FuncOp returning quantum_node results
        qnode_call_results = call_op.results
        return_op = func.ReturnOp(*qnode_call_results)
        outer_fn.body.block.add_op(return_op)


def _get_parent_module(op: func.FuncOp) -> builtin.ModuleOp:
    """Get the first ancestral builtin.ModuleOp op of a given func.func op."""
    _op: Operation | None = op
    while _op := _op.parent_op():
        if isinstance(_op, builtin.ModuleOp):
            break
        if _op is None:
            raise CompileError(f"{op} does not belong to a module.")

    assert isinstance(_op, builtin.ModuleOp)
    return _op
