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
from enum import Enum
from pathlib import Path
from typing import ContextManager, List, Optional, Set

from jax._src.interpreters.partial_eval import DynamicJaxprTrace
from jax.core import find_top_trace, set_current_trace, take_current_trace
from pennylane.queuing import QueuingManager

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
    _am_inside_accelerate: int = 0

    def __enter__(self):
        AccelerateContext._am_inside_accelerate += 1

    def __exit__(self, _exc_type, _exc, _exc_tb):
        AccelerateContext._am_inside_accelerate -= 1

    @staticmethod
    def am_inside_accelerate():
        return AccelerateContext._am_inside_accelerate > 0


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


class EvaluationContext:
    """Utility context managing class keeping track of various modes of Catalyst executions.

    It is used to determine whether the program is currently tracing or not and if so, tracking the
    tracing contexts. Contexts can be nested.
    """

    _mode_stack: List[EvaluationMode] = []
    _mlir_plugins: Set[Path] = set()

    @debug_logger_init
    def __init__(self, mode: EvaluationMode):
        """Initialise a new instance of the Evaluation context.
        Args:
            mode: Evaluation mode of this instance
        """
        self.mode = mode
        self._ctx = None

    @classmethod
    def add_plugin(cls, plugin: Path):
        """Add an MLIR plugin to the set of MLIR plugins encountered in the
        program"""
        cls._mlir_plugins.add(plugin)

    @classmethod
    def get_plugins(cls):
        """Get and reset all plugins encountered during the trace of the
        program"""
        retval = cls._mlir_plugins
        cls._mlir_plugins = set()
        return retval

    @classmethod
    @contextmanager
    def _create_tracing_context(cls, mode):
        cls._mode_stack.append(mode)
        try:
            yield
        finally:
            cls._mode_stack.pop()

    @classmethod
    @contextmanager
    def _create_interpretation_context(cls):
        cls._mode_stack.append(EvaluationMode.INTERPRETATION)
        try:
            yield None
        finally:
            cls._mode_stack.pop()

    @classmethod
    @contextmanager
    def frame_tracing_context(
        cls, trace: Optional[DynamicJaxprTrace] = None, debug_info=None
    ) -> ContextManager[DynamicJaxprTrace]:
        """Start a new JAX tracing frame, e.g. to trace a region of some
        :class:`~.jax_tracer.HybridOp`. Not applicable in non-tracing evaluation modes."""
        with take_current_trace():
            if trace is not None:
                new_trace = trace
            else:
                new_trace = DynamicJaxprTrace(debug_info)

        with set_current_trace(new_trace):
            try:
                yield new_trace
            finally:
                del new_trace

    @classmethod
    def get_current_trace(cls, hint=None):
        """Return the current JAX trace, raise an exception if not in tracing mode."""
        msg = f"{hint or 'catalyst functions'} can only be used from within @qjit decorated code."
        EvaluationContext.check_is_tracing(msg)
        with take_current_trace() as current_trace:
            return current_trace

    def __enter__(self):
        if self.mode in [EvaluationMode.QUANTUM_COMPILATION, EvaluationMode.CLASSICAL_COMPILATION]:
            self._ctx = self._create_tracing_context(self.mode)
        else:
            assert self.mode in {EvaluationMode.INTERPRETATION}, f"Unknown mode {self.mode}"
            self._ctx = self._create_interpretation_context()
        return self._ctx.__enter__()

    def __exit__(self, *args, **kwargs):
        self._ctx.__exit__(*args, **kwargs)

    @classmethod
    def get_evaluation_mode(cls) -> EvaluationMode:
        """Return the name of the evaluation mode, paired with tracing context if applicable"""
        if not EvaluationContext._mode_stack:
            return EvaluationMode.INTERPRETATION
        return cls._mode_stack[-1]

    @classmethod
    def get_mode(cls):
        """Return the name of current evaluation mode."""
        return cls.get_evaluation_mode()

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
