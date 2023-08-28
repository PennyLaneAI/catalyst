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

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, Union

from jax._src.core import ClosedJaxpr, JaxprEqn
from jax._src.core import MainTrace as JaxMainTrace
from jax._src.core import ShapedArray, cur_sublevel, new_base_main
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTrace,
    DynamicJaxprTracer,
    JaxprStackFrame,
    extend_jaxpr_stack,
)
from jax._src.source_info_util import current as jax_current
from jax._src.source_info_util import reset_name_stack

from catalyst.utils.exceptions import CompileError
from catalyst.utils.jax_extras import new_main2


class EvaluationMode(Enum):
    INTERPRETATION = 0
    CLASSICAL_COMPILATION = 2
    QUANTUM_COMPILATION = 1


@dataclass
class JaxTracingContext:
    main: JaxMainTrace
    frames: Dict[DynamicJaxprTrace, JaxprStackFrame]
    mains: Dict[DynamicJaxprTrace, JaxMainTrace]
    trace: Optional[DynamicJaxprTrace]

    def __init__(self, main: JaxMainTrace):
        self.main, self.frames, self.mains, self.trace = main, {}, {}, None


class EvaluationContext:
    """Utility class used for tracing.

    It is used to determine whether the program is currently tracing or not.
    """

    _tracing_stack: List[Tuple[EvaluationMode, Optional[JaxTracingContext]]] = []

    def __init__(self, mode: EvaluationMode):
        self.mode = mode
        self.ctx = None

    @classmethod
    @contextmanager
    def create_tracing_context(cls, mode) -> ContextManager[JaxTracingContext]:
        with new_base_main(DynamicJaxprTrace, dynamic=True) as main:
            main.jaxpr_stack = ()
            cls._tracing_stack.append((mode, JaxTracingContext(main)))
            try:
                yield cls._tracing_stack[-1][1]
            finally:
                cls._tracing_stack.pop()

    @classmethod
    @contextmanager
    def create_interpretation_context(cls) -> ContextManager[JaxTracingContext]:
        cls._tracing_stack.append((EvaluationMode.INTERPRETATION, None))
        try:
            yield cls._tracing_stack[-1][1]
        finally:
            cls._tracing_stack.pop()

    @classmethod
    @contextmanager
    def frame_tracing_context(
        cls, ctx: JaxTracingContext, trace: Optional[DynamicJaxprTrace] = None
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
    def get_main_tracing_context(cls, hint=None) -> JaxTracingContext:
        """Checks a number of tracing conditions and return the JaxTracingContext"""
        msg = f"{hint or 'catalyst functions'} can only be used from within @qjit decorated code."
        EvaluationContext.check_is_tracing(msg)
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
        if not EvaluationContext._tracing_stack:
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
        return cls.get_mode() in [
            EvaluationMode.CLASSICAL_COMPILATION,
            EvaluationMode.QUANTUM_COMPILATION,
        ]

    @classmethod
    def check_modes(cls, modes, msg):
        """Assert if the execution is currently not being traced.

        Raises: CompileError
        """
        if cls.get_mode() not in modes:
            raise CompileError(msg)

    @classmethod
    def check_is_quantum_tracing(cls, msg):
        cls.check_modes([EvaluationMode.QUANTUM_COMPILATION], msg)

    @classmethod
    def check_is_classical_tracing(cls, msg):
        cls.check_modes([EvaluationMode.CLASSICAL_COMPILATION], msg)

    @classmethod
    def check_is_tracing(cls, msg):
        cls.check_modes(
            [EvaluationMode.CLASSICAL_COMPILATION, EvaluationMode.QUANTUM_COMPILATION], msg
        )

    @classmethod
    def check_is_not_tracing(cls, msg):
        """Assert if the execution is currently being traced.

        Raises: CompileError
        """
        if cls.is_tracing():
            raise CompileError(msg)
