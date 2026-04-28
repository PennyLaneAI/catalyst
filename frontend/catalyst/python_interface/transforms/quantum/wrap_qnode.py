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

from dataclasses import dataclass

from pennylane.exceptions import CompileError
from xdsl import context, passes, pattern_rewriter
from xdsl.dialects import builtin, func
from xdsl.ir import Operation
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import InsertPoint


def get_call_op(qnode: func.FuncOp):
    """Get the CallOp in another module function that calls this quantum_node. Postprocessing
    can be injected to act on the output of that CallOp.
    """

    module = _get_parent_module(qnode)
    qnode_name = qnode.sym_name.data
    all_call_ops = [op for op in module.body.walk() if isinstance(op, func.CallOp)]
    qnode_call_op = [op for op in all_call_ops if qnode_name in op.callee.string_value()]
    if not len(qnode_call_op) == 1:
        raise CompileError(
            f"Expected only one call_op for {qnode_name}, but received {len(qnode_call_op)}"
        )

    return qnode_call_op[0]


@dataclass(frozen=True)
class WrapQNodePass(passes.ModulePass):
    """This pass is a utility intended to be used with passes that update a quantum.node
    and add post-processing. The pass wraps each quantum.node in the program with a classical
    function that calls the quantum.node. Postprocessing can then be inserted after the call
    in application of a subsequent Pattern.

        Args:
            pass_str (str): The string used to label the QNode. This can be used to provide
                information in the IR about which pass any classical post-processing relates to.

    **Example**

    Let's say we want to add a pass, SomePass, that modifies a QNode in a way that requires
    post-processing. We can apply the WrapQNodePass before the Pattern for SomePase to wrap
    the QNodes in classical functions that call them:

    .. code-block:: python

        @dataclass(frozen=True)
        class SomePass(passes.ModulePass):

            name = "some-name"

            def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:

                WrapQNodePass("some_name").apply(_ctx, op)

                pattern_rewriter.PatternRewriteWalker(
                    SomePassPattern(),
                    apply_recursively=False,
                ).rewrite_module(op)

    Within SomePassPattern, post-processing can be injected after the CallOp inside
    the classical function. Once the WrapQNodePass is applied, the CallOp for a
    quantum.node FuncOp can be accessed using the get_call_op function.
    """

    name = "wrap-qnode"

    pass_str: str

    def apply(self, _ctx: context.Context, op: builtin.ModuleOp) -> None:
        """."""

        pattern_rewriter.PatternRewriteWalker(
            WrapQNodePattern(pass_str=self.pass_str),
            apply_recursively=False,
        ).rewrite_module(op)


class WrapQNodePattern(RewritePattern):
    """RewritePattern wrapping a quantum.node inside a classical function that calls the
    quantum.node. Postprocessing can then be inserted after the call by Patterns that add
    post-processing.

    The new classical function will have the name, inputs and output types as the QNode,
    so CallOps that previously called the QNode will now call the wrapper function. The
    QNode will have a string appended to its name to differentiate it from the wrapper
    function.

        Args:
            pass_str: The string used to label the QNode.

    **Example**

    .. code-block:: mlir

    module {
      func.func @my_circuit(%angle: f64) -> !quantum.bit {
        %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
        %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
        %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
        %3 = quantum.custom "RX"(%angle) %2 : !quantum.bit
        return %3 : !quantum.bit
      }
    }

    """

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
        postprocessing function is given the original quantum.node name.
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
