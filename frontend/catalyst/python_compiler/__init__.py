from typing import Callable

from dataclasses import dataclass
from xdsl import context, passes
from xdsl.utils import parse_pipeline
from xdsl.dialects import builtin

@dataclass(frozen=True)
# All passes inherit from passes.ModulePass
class PrintModule(passes.ModulePass):
    # All passes require a name field
    name = "print"

    # All passes require an apply method with this signature.
    def apply(self, ctx: context.MLContext, module: builtin.ModuleOp) -> None:
        print("Hello from inside the pass\n", module)
