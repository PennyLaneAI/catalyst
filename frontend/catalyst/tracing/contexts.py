# Copyright 2022-2024 Xanadu Quantum Technologies Inc.

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
This module provides context classes to manage and query Catalyst's and JAX's tracing state.
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import ContextManager, Dict, List, Optional, Tuple

from jax._src.core import MainTrace as JaxMainTrace
from jax._src.core import cur_sublevel, new_base_main
from jax._src.interpreters.partial_eval import (
    DynamicJaxprTrace,
    JaxprStackFrame,
    extend_jaxpr_stack,
)
from jax._src.source_info_util import reset_name_stack
from jax.core import find_top_trace
from pennylane.queuing import QueuingManager

from catalyst.jax_extras import new_dynamic_main2
from catalyst.logging import debug_logger_init
from catalyst.utils.exceptions import CompileError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class GradContext:
    """This class tells us the level of nestedness of grad."""

    # Conceptually:
    #
    #   _grad_stack : List[bool] = []
    #
    # is a class member variable that reflects the level of
    # nested-ness of the grad operation. If we have the following code:
    #
    #   grad(fun)(x)
    #
    # and we are inside fun, then List _grad_stack will contain [True]
    # if we have:
    #
    #  grad(grad(fun))(x)
    #
    # and we are tracing fun, then _grad_stack will contain [True, True]
    #
    # I say conceptually because you can already tell that we don't need
    # anything besides an integer that gets incremented and decremented
    # each time you enter a grad or jacobian operation.
    #
    # The reason why we started with a stack is because I think it makes more sense.
    #
    # You might also ask, why do we need keep track of the order of the derivative?
    # Isn't it enough to know that we are inside the grad context or outside?
    # It is when you are only limited to the first order derivative, but once
    # you have a higher order derivatives, it will be important to know whether
    # you need the derivative of the derivative...
    #
    # The final question to be asked here is whether we need a new stack for this
    # context or whether we should reuse the old one.
    # I think we should keep it simple and this is simpler.

    _grad_stack: int = 0
    # This message will be used in an assertion because it is not expected
    # to be a user facing error ever.

    def __init__(self, peel=False):
        """Peel is useful when we want to temporarily create a context
        where we peel one order of the derivative"""
        self.peel = peel

    def __enter__(self, peel=False):
        _ = GradContext._pop() if self.peel else GradContext._push()

    def __exit__(self, _exc_type, _exc, _exc_tb):
        _ = GradContext._push() if self.peel else GradContext._pop()

    @staticmethod
    def am_inside_grad():
        """Return true if we are currently tracing inside a grad operation."""
        return GradContext._peek() > 0

    @staticmethod
    def _pop():
        retval = GradContext._peek()
        msg = "This is an impossible state. "
        msg += "One cannot derive to derive to a negative order / integrate"
        assert retval > 0, msg
        GradContext._grad_stack -= 1
        return retval

    @staticmethod
    def _push():
        GradContext._grad_stack += 1

    @staticmethod
    def _peek():
        return GradContext._grad_stack


class AccelerateContext:
    _am_inside_accelerate: bool = False

    def __enter__(self):
        AccelerateContext._am_inside_accelerate = True

    def __exit__(self, _exc_type, _exc, _exc_tb):
        AccelerateContext._am_inside_accelerate = False

    @staticmethod
    def am_inside_accelerate():
        return AccelerateContext._am_inside_accelerate


class EvaluationMode(Enum):
    """Enumerate the evaluation modes supported by Catalyst:
    INTERPRETATION - native Python execution of a Catalyst program
    QUANTUM_COMPILATION - JAX tracing followed by MLIR compilation in the presence of quantum
                          instructions
    CLASSICAL_COMPILATION - JAX tracing followed JAX compilation of classical-only Catalyst
                            programs.
    """

    INTERPRETATION = 0
    QUANTUM_COMPILATION = 1
    CLASSICAL_COMPILATION = 2


@dataclass
class JaxTracingContext:
    """JAX tracing context supporting nested quantum operations. Keeps track of the re-entrable
    tracing frames. The tracing algorithm visits these frames several times: first during the
    classical tracing, then during the quantum tracing and also during the optional transformations.

    Args:
        main: Base JAX tracing data structure.
        frames: JAX tracing frames; holding the JAXPR equations.
        mains: Secondary JAX tracing structrures. Each structure has one frame and
               corresponds to a :class:`~.jax_tracer.HybridOpRegion`
        trace: Current JAX trace object.
    """

    main: JaxMainTrace
    frames: Dict[DynamicJaxprTrace, JaxprStackFrame]
    mains: Dict[DynamicJaxprTrace, JaxMainTrace]
    trace: Optional[DynamicJaxprTrace]

    @debug_logger_init
    def __init__(self, main: JaxMainTrace):
        self.main, self.frames, self.mains, self.trace = main, {}, {}, None


class EvaluationContext:
    """Utility context managing class keeping track of various modes of Catalyst executions.

    It is used to determine whether the program is currently tracing or not and if so, tracking the
    tracing contexts. Contexts can be nested.
    """

    _tracing_stack: List[Tuple[EvaluationMode, Optional[JaxTracingContext]]] = []

    @debug_logger_init
    def __init__(self, mode: EvaluationMode):
        """Initialise a new instance of the Evaluation context.
        Args:
            mode: Evaluation mode of this instance
        """
        self.mode = mode
        self.ctx = None

    @classmethod
    @contextmanager
    def _create_tracing_context(cls, mode) -> ContextManager[JaxTracingContext]:
        with new_base_main(DynamicJaxprTrace, dynamic=True) as main:
            main.jaxpr_stack = ()
            cls._tracing_stack.append((mode, JaxTracingContext(main)))
            try:
                yield cls._tracing_stack[-1][1]
            finally:
                cls._tracing_stack.pop()

    @classmethod
    @contextmanager
    def _create_interpretation_context(cls) -> ContextManager[JaxTracingContext]:
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
        """Start a new JAX tracing frame, e.g. to trace a region of some
        :class:`~.jax_tracer.HybridOp`. Not applicable in non-tracing evaluation modes."""
        assert ctx is cls._tracing_stack[-1][1], f"{ctx=}"
        main = ctx.mains[trace] if trace is not None else None
        with new_dynamic_main2(DynamicJaxprTrace, main=main) as nmain:
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
        """Return the current JAX tracing context, raise an exception if not in tracing mode."""
        msg = f"{hint or 'catalyst functions'} can only be used from within @qjit decorated code."
        EvaluationContext.check_is_tracing(msg)
        return cls._tracing_stack[-1][1]

    def __enter__(self):
        if self.mode in [EvaluationMode.QUANTUM_COMPILATION, EvaluationMode.CLASSICAL_COMPILATION]:
            self.ctx = self._create_tracing_context(self.mode)
        else:
            assert self.mode in {EvaluationMode.INTERPRETATION}, f"Unknown mode {self.mode}"
            self.ctx = self._create_interpretation_context()
        return self.ctx.__enter__()

    def __exit__(self, *args, **kwargs):
        self.ctx.__exit__(*args, **kwargs)

    @classmethod
    def get_evaluation_mode(cls) -> Tuple[EvaluationMode, Optional[JaxTracingContext]]:
        """Return the name of the evaluation mode, paired with tracing context if applicable"""
        if not EvaluationContext._tracing_stack:
            return (EvaluationMode.INTERPRETATION, None)
        return cls._tracing_stack[-1]

    @classmethod
    def get_mode(cls):
        """Return the name of current evaluation mode."""
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
    def is_quantum_tracing(cls):
        """Returns true or false depending on whether the execution is currently being
        traced.
        """
        return cls.get_mode() == EvaluationMode.QUANTUM_COMPILATION

    @classmethod
    def check_modes(cls, modes, msg):
        """Asserts if the execution mode is not among the expected ``modes``.

        Raises: CompileError
        """
        if cls.get_mode() not in modes:
            raise CompileError(msg)

    @classmethod
    def check_is_quantum_tracing(cls, msg):
        """Asserts if the current evaluation mode is quantum tracing.

        Raises: CompileError
        """
        cls.check_modes([EvaluationMode.QUANTUM_COMPILATION], msg)

    @classmethod
    def check_is_classical_tracing(cls, msg):
        """Asserts if the current evaluation mode is classical tracing.

        Raises: CompileError
        """
        cls.check_modes([EvaluationMode.CLASSICAL_COMPILATION], msg)

    @classmethod
    def check_is_tracing(cls, msg):
        """Asserts if the current evaluation mode is not a tracing.

        Raises: CompileError
        """
        cls.check_modes(
            [EvaluationMode.CLASSICAL_COMPILATION, EvaluationMode.QUANTUM_COMPILATION], msg
        )

    @classmethod
    def check_is_not_tracing(cls, msg):
        """Asserts if the current execution mode is a tracing.

        Raises: CompileError
        """
        if cls.is_tracing():
            raise CompileError(msg)

    @classmethod
    def find_jaxpr_frame(cls, *args):
        """Obtain the current JAXPR frame, in which primitives are being inserted.

        Raises: CompileError
        """
        cls.check_is_tracing("No JAXPR frames exist outside a tracing context.")
        return find_top_trace(args).frame

    @classmethod
    def find_quantum_queue(cls):
        """Obtain the current quantum queuing context, in which operations are being inserted.

        Raises: CompileError
        """
        cls.check_is_quantum_tracing("No quantum queueing context found.")

        queuing_context = QueuingManager.active_context()
        assert queuing_context is not None

        return queuing_context
