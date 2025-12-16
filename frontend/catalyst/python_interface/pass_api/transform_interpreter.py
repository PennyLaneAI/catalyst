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

# pylint: disable=line-too-long
"""Custom Transform Dialect Interpreter Pass

Differs from xDSL's upstream implementation by allowing passes
to be passed in as options and for the pipeline to have a callback and apply it
after every pass is run. The callback differs from how xDSL callback mechanism
is integrated into the PassPipeline object since PassPipeline only runs
if there are more than two passes. Here we are running one pass at a time
which will prevent the callback from being called.


See here (link valid with xDSL 0.46): https://github.com/xdslproject/xdsl/blob/334492e660b1726bc661efc7afb927e74bac48f4/xdsl/passes.py#L211-L222
"""

import io
from collections.abc import Callable, Sequence
from typing import Any

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.dialects.transform import ApplyRegisteredPassOp, NamedSequenceOp
from xdsl.interpreter import Interpreter, PythonValues, impl, register_impls
from xdsl.interpreters.transform import TransformFunctions
from xdsl.ir import Attribute
from xdsl.parser import Parser
from xdsl.passes import ModulePass, PassPipeline
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import PassFailedException
from xdsl.utils.hints import isa

from catalyst.compiler import _quantum_opt


@register_impls
class TransformFunctionsExt(TransformFunctions):
    """
    Unlike the implementation available in xDSL, this implementation overrides
    the semantics of the `transform.apply_registered_pass` operation by
    first always attempting to apply the xDSL pass, but if it isn't found
    then it will try to run this pass in Catalyst.
    """

    def __init__(self, ctx, passes, callback=None):
        super().__init__(ctx, passes)
        # The signature of the callback function is assumed to be
        # def callback(previous_pass: ModulePass, module: ModuleOp, next_pass: ModulePass, pass_level=None) -> None
        self.callback = callback
        self.pass_level = 0

    def _pre_pass_callback(self, compilation_pass, module):
        """Callback wrapper to run the callback function before the pass."""
        if not self.callback:
            return
        if self.pass_level == 0:
            # Since this is the first pass, there is no previous pass
            self.callback(None, module, compilation_pass, pass_level=0)

    def _post_pass_callback(self, compilation_pass, module):
        """Increment level and run callback if defined."""
        if not self.callback:
            return
        self.pass_level += 1
        self.callback(compilation_pass, module, None, pass_level=self.pass_level)

    @impl(ApplyRegisteredPassOp)
    def run_apply_registered_pass_op(
        self,
        _interpreter: Interpreter,
        op: ApplyRegisteredPassOp,
        args: PythonValues,
    ) -> PythonValues:
        """Try to run the pass in xDSL, if not found then run it in Catalyst."""

        pass_name = op.pass_name.data
        module = args[0]

        # ---- xDSL path ----
        if pass_name in self.passes:
            pass_class = self.passes[pass_name]()
            pass_instance = pass_class(**op.options.data)
            pipeline = PassPipeline((pass_instance,))
            self._pre_pass_callback(pass_instance, module)
            pipeline.apply(self.ctx, module)
            self._post_pass_callback(pass_instance, module)
            return (module,)

        # ---- Catalyst path ----
        buffer = io.StringIO()
        Printer(stream=buffer, print_generic_format=True).print_op(module)
        schedule = _create_schedule([op])
        self._pre_pass_callback(pass_name, module)
        modified = _quantum_opt(*schedule, "-mlir-print-op-generic", stdin=buffer.getvalue())

        data = Parser(self.ctx, modified).parse_module()
        rewriter = Rewriter()
        rewriter.replace_op(module, data)
        self._post_pass_callback(pass_name, data)
        return (data,)


class TransformInterpreterPass(ModulePass):
    """Transform dialect interpreter"""

    passes: dict[str, Callable[[], type[ModulePass]]]
    name = "transform-interpreter"
    callback: Callable[[ModulePass, builtin.ModuleOp, ModulePass], None] | None = None

    entry_point: str = "__transform_main"

    def __init__(self, passes, callback):
        self.passes = passes
        self.callback = callback

    @staticmethod
    def find_transform_entry_point(root: builtin.ModuleOp, entry_point: str) -> NamedSequenceOp:
        """Find the entry point of the program"""
        for op in root.walk():
            if isinstance(op, NamedSequenceOp) and op.sym_name.data == entry_point:
                return op
        raise PassFailedException(  # pragma: no cover
            f"{root} could not find a nested named sequence with name: {entry_point}"
        )

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        """Run the interpreter with op."""
        schedule = TransformInterpreterPass.find_transform_entry_point(op, self.entry_point)
        interpreter = Interpreter(op)
        interpreter.register_implementations(TransformFunctionsExt(ctx, self.passes, self.callback))
        schedule.parent_op().detach()
        if self.callback:
            self.callback(None, op, None, pass_level=0)
        interpreter.call_op(schedule, (op,))


def _create_schedule(pass_ops: Sequence[ApplyRegisteredPassOp]) -> list[str]:
    """Create a pass schedule for applying MLIR pass via CLI flags.

    For a pass with options, the corresponding CLI flag will be the following:

    .. code-block::

        --my-pass="arg1=val1 arg2=val2 ..."

    Args:
        pass_ops (Sequence[xdsl.dialects.transform.ApplyRegisteredPassOp]): The
            passes to schedule

    Returns:
        list[str]: A list containing strings that correspond to the CLI flags
        for the specified passes
    """
    schedule: list[str] = []

    for op in pass_ops:
        pass_name = op.pass_name.data
        pass_options = op.options.data

        if not pass_options:
            schedule.append(f"--{pass_name}")
            continue

        cli_options = []
        for opt, val in pass_options.items():
            cli_options.append(f"{opt}={_get_cli_option_from_attr(val)}")
        cli_options = " ".join(cli_options)

        cli_pass = f"--{pass_name}={cli_options}"
        schedule.append(cli_pass)

    return schedule


def _get_cli_option_from_attr(val: Attribute) -> Any:
    """Convert an xDSL attribute corresponding to a pass option value into a valid
    CLI option value."""
    cli_val = None

    match val:
        case builtin.IntegerAttr():
            if _is_bool_attr(val):
                # Python booleans' repr is capitalized so we use strings
                cli_val = "true" if val.value.data else "false"
            else:
                cli_val = val.value.data

        case builtin.FloatAttr():
            cli_val = val.value.data

        case builtin.StringAttr():
            cli_val = val.data

        case builtin.ArrayAttr():
            cli_val = ",".join([_get_cli_option_from_attr(attr) for attr in val.data])

        case builtin.DictionaryAttr():
            mapping = []
            for k, v in val.data.items():
                mapping.append(f"{k}={_get_cli_option_from_attr(v)}")
            cli_val = f"{{{' '.join(mapping)}}}"

        case _:
            raise ValueError(f"Unsupported option type {val}.")

    return cli_val


def _is_bool_attr(val: builtin.IntegerAttr) -> bool:
    """Check if an IntegerAttr corresponds to a boolean by checking its width."""
    # Boolean attributes will be of Integer type with bitwidth == 1
    return isinstance(val.type, builtin.IntegerType) and val.type.width.data == 1
