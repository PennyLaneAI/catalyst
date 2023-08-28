# Copyright 2022-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tracing module.
"""

from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, Union
from catalyst.utils.exceptions import CompileError
from enum import Enum
from contextlib import contextmanager
from dataclasses import dataclass

from jax._src.core import (
    cur_sublevel,
    new_base_main,
    ClosedJaxpr,
    JaxprEqn,
    MainTrace as JaxMainTrace,
    ShapedArray,
)
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    JaxprStackFrame,
    extend_jaxpr_stack,
)
from catalyst.utils.jax_extras import (
    new_main2,
)
from jax._src.source_info_util import (reset_name_stack, current as jax_current)

class EvaluationMode(Enum):
    INTERPRETATION = 0
    CLASSICAL_COMPILATION = 2
    QUANTUM_COMPILATION = 1


@dataclass
class MainTracingContext:
    main: JaxMainTrace
    frames: Dict[DynamicJaxprTrace, JaxprStackFrame]
    mains: Dict[DynamicJaxprTrace, JaxMainTrace]
    trace: Optional[DynamicJaxprTrace]

    def __init__(self, main: JaxMainTrace):
        self.main, self.frames, self.mains, self.trace = main, {}, {}, None


class TracingContext:
    """Utility class used for tracing.

    It is used to determine whether the program is currently tracing or not.
    """

    _tracing_stack:List[Tuple[EvaluationMode, Optional[MainTracingContext]]] = []

    def __init__(self, mode:EvaluationMode):
        self.mode = mode
        self.ctx = None

    @classmethod
    @contextmanager
    def create_tracing_context(cls, mode) -> ContextManager[MainTracingContext]:
        with new_base_main(DynamicJaxprTrace, dynamic=True) as main:
            main.jaxpr_stack = ()
            cls._tracing_stack.append((mode, MainTracingContext(main)))
            try:
                yield cls._tracing_stack[-1][1]
            finally:
                cls._tracing_stack.pop()

    @classmethod
    @contextmanager
    def create_interpretation_context(cls) -> ContextManager[MainTracingContext]:
        cls._tracing_stack.append((EvaluationMode.INTERPRETATION, None))
        try:
            yield cls._tracing_stack[-1][1]
        finally:
            cls._tracing_stack.pop()

    @classmethod
    @contextmanager
    def frame_tracing_context(
        cls, ctx, trace: Optional[DynamicJaxprTrace] = None
    ) -> ContextManager[DynamicJaxprTrace]:
        assert ctx is cls._tracing_stack[-1][1], f"{ctx=}"
        main = ctx.mains[trace] if trace is not None else None
        with new_main2(DynamicJaxprTrace, dynamic=True, main=main) as nmain:
            nmain.jaxpr_stack = ()
            frame = JaxprStackFrame() if trace is None else ctx.frames[trace]
            with extend_jaxpr_stack(nmain, frame), reset_name_stack():
                parent_trace = ctx.trace
                ctx.trace = DynamicJaxprTrace(nmain, cur_sublevel()) if trace is None else trace
                ctx.frames[ctx.trace] = frame
                ctx.mains[ctx.trace] = nmain
                try:
                    yield ctx.trace
                finally:
                    ctx.trace = parent_trace

    @classmethod
    def get_main_tracing_context(cls, hint=None) -> MainTracingContext:
        """Checks a number of tracing conditions and return the MainTracingContext"""
        msg = f"{hint or 'catalyst functions'} can only be used from within @qjit decorated code."
        TracingContext.check_is_tracing(msg)
        if len(cls._tracing_stack) == 0:
            raise CompileError(f"{hint} can only be used from within a qml.qnode.")
        return cls._tracing_stack[-1][1]

    def __enter__(self):
        if self.mode in [EvaluationMode.QUANTUM_COMPILATION, EvaluationMode.CLASSICAL_COMPILATION]:
            self.ctx = self.create_tracing_context(self.mode)
        else:
            assert self.mode in {EvaluationMode.INTERPRETATION}, f"Unknown mode {self.mode}"
            self.ctx = self.create_interpretation_context()
        return self.ctx.__enter__()

    def __exit__(self, *args, **kwargs):
        self.ctx.__exit__(*args, **kwargs)

    @classmethod
    def get_evaluation_mode(cls):
        if not TracingContext._tracing_stack:
            return (EvaluationMode.INTERPRETATION, None)
        return cls._tracing_stack[-1]

    @classmethod
    def get_mode(cls):
        return cls.get_evaluation_mode()[0]

    @classmethod
    def is_tracing(cls):
        """Returns true or false depending on whether the execution is currently being
        traced.
        """
        return cls.get_mode() in [EvaluationMode.CLASSICAL_COMPILATION,
                                  EvaluationMode.QUANTUM_COMPILATION]

    @staticmethod
    def check_is_tracing(msg):
        """Assert if the execution is currently not being traced.

        Raises: CompileError
        """
        if not TracingContext.is_tracing():
            raise CompileError(msg)

    @staticmethod
    def check_is_not_tracing(msg):
        """Assert if the execution is currently being traced.

        Raises: CompileError
        """
        if TracingContext.is_tracing():
            raise CompileError(msg)

