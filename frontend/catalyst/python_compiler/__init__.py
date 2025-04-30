from typing import Callable

from dataclasses import dataclass
from xdsl import context, passes
from xdsl.utils import parse_pipeline
from xdsl.dialects import builtin

from dataclasses import dataclass

from xdsl.dialects import builtin, transform
from xdsl.interpreter import Interpreter
from xdsl.interpreters.transform import TransformFunctions
from xdsl.passes import Context, ModulePass
from xdsl.utils.exceptions import PassFailedException

from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    TerminatorValue,
    impl,
    impl_callable,
    impl_terminator,
    register_impls,
)

@register_impls
class TransformFunctionsExt(TransformFunctions):
    ctx: Context
    passes: dict[str, Callable[[], type[ModulePass]]]

    def __init__(
        self, ctx: Context, available_passes: dict[str, Callable[[], type[ModulePass]]]
    ):
        self.ctx = ctx
        self.passes = available_passes

    @impl(transform.ApplyRegisteredPassOp)
    def run_apply_registered_pass_op(
        self,
        interpreter: Interpreter,
        op: transform.ApplyRegisteredPassOp,
        args: PythonValues,
    ) -> PythonValues:
        pass_name = op.pass_name.data
        requested_by_user = passes.PipelinePass.build_pipeline_tuples(
            self.passes, parse_pipeline.parse_pipeline(pass_name)
        )
        # TODO: Switch between catalyst and xDSL

        schedule = tuple(
            pass_type.from_pass_spec(spec) for pass_type, spec in requested_by_user
        )
        pipeline = passes.PipelinePass(schedule)
        pipeline.apply(self.ctx, args[0])
        return (args[0],)

from xdsl.transforms import get_all_passes

updated_passes = get_all_passes()

def register_pass(name, _callable):
    global updated_passes
    updated_passes[name] = _callable

@dataclass(frozen=True)
class TransformInterpreterPass(ModulePass):
    """Transform dialect interpreter"""

    name = "transform-interpreter"

    entry_point: str = "__transform_main"

    @staticmethod
    def find_transform_entry_point(
        root: builtin.ModuleOp, entry_point: str
    ) -> transform.NamedSequenceOp:
        for op in root.walk():
            if (
                isinstance(op, transform.NamedSequenceOp)
                and op.sym_name.data == entry_point
            ):
                return op
        raise PassFailedException(
            f"{root} could not find a nested named sequence with name: {entry_point}"
        )

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        schedule = TransformInterpreterPass.find_transform_entry_point(
            op, self.entry_point
        )
        interpreter = Interpreter(op)
        global updated_passes
        interpreter.register_implementations(TransformFunctionsExt(ctx, updated_passes))
        interpreter.call_op(schedule, (op,))

@dataclass(frozen=True)
class ApplyTransformSequence(passes.ModulePass):
    name = "apply-transform-sequence"

    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        nested_modules = []
        for region in module.regions:
          for block in region.blocks:
              for op in block.ops:
                  if isinstance(op, builtin.ModuleOp):
                      nested_modules.append(op)

        pipeline = passes.PipelinePass((TransformInterpreterPass(),))
        for op in nested_modules:
            pipeline.apply(ctx, op)

        for op in nested_modules:
            for region in op.regions:
                for block in region.blocks:
                    for op in block.ops:
                        if isinstance(op, builtin.ModuleOp) and op.get_attr_or_prop("transform.with_named_sequence"):
                            block.erase_op(op)

