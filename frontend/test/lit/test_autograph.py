# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the AutoGraph source-to-source transformation feature."""

# RUN: %PYTHON %s | FileCheck %s

from catalyst import qjit
from catalyst.ag_utils import AutoGraphError, print_code
from catalyst.autograph import autograph


# CHECK-LABEL: def while_simple
@autograph
def while_simple(x: float):
    """Test a simple while-loop statemnt."""

    # CHECK:   def loop_body
    # CHECK:   def loop_test
    # CHECK:   ag__.while_stmt(loop_test, loop_body
    while x < 1.0:
        x *= 1.5
    return x


print_code(while_simple)


# -----


# CHECK-LABEL: def while_default_jax
@qjit(autograph=True)
def while_default_jax(a: int):
    """Checks that failure during the while-loop tracing is detected and the fallback unrolling is
    executed."""
    i = 0
    acc = 0
    # CHECK: qwhile
    while i < 5:
        acc += a
        i += 1
    return acc


print("def while_default_jax")
print(while_default_jax.jaxpr)

# -----


class Failing:
    """Test class that emulates failures in user-code"""

    triggered = False

    def __init__(self, ref):
        self.ref = ref

    @property
    def val(self):
        """Get a reference to a variable or fail if programmed so."""
        # pylint: disable=broad-exception-raised
        if not Failing.triggered:
            Failing.triggered = True
            raise Exception("Emulated failure")
        return self.ref


# CHECK-LABEL: def while_fallback_jax
@qjit(autograph=True)
def while_fallback_jax(a: int):
    """Checks that failure during the while-loop tracing is detected and the fallback unrolling is
    executed."""
    i = 0
    acc = 0
    # CHECK: add
    # CHECK: add
    # CHECK: add
    # CHECK: add
    # CHECK: add
    while Failing(i).val < 5:
        acc += a
        i += 1
    return acc


print("def while_fallback_jax")
print(while_fallback_jax.jaxpr)


# -----


# CHECK-LABEL: def if_simple
@autograph
def if_simple(x: float):
    """Test a simple conditional with a single branch."""

    # CHECK:   def if_body():
    # CHECK:       pass
    if x < 3:
        pass
    # CHECK:   def else_body():
    # CHECK:       pass

    # CHECK:   ag__.if_stmt(x < 3, if_body, else_body, get_state, set_state, (), 0)

    return x


print_code(if_simple)

# -----


# CHECK-LABEL: def if_else
@autograph
def if_else(x: float):
    """Test a simple conditional with two branches."""

    # CHECK:   def if_body():
    # CHECK:       pass
    if x < 3:
        pass
    # CHECK:   def else_body():
    # CHECK:       pass
    else:
        pass

    # CHECK:   ag__.if_stmt(x < 3, if_body, else_body, get_state, set_state, (), 0)

    return x


print_code(if_else)

# -----


# CHECK-LABEL: def if_assign
@autograph
def if_assign(x: float):
    """Test a conditional creates a new variable."""

    # CHECK:   def if_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 4
    if x < 3:
        y = 4
    # CHECK:   def else_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 5
    else:
        y = 5

    # CHECK:   ag__.if_stmt(x < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   return y
    return y


print_code(if_assign)

# -----


try:

    @qjit  # needed to trigger Catalyst type checks during tracing
    @autograph
    def if_assign_type_mismatch(x: float):
        """Verify error from a conditional that doesn't produce the same type across branches."""

        if x < 3:
            y = 4.0
        else:
            y = 5

        return y

except TypeError as e:
    # CHECK:   Conditional requires consistent return types across all branches
    print(e)

# -----


try:

    @qjit  # needed to trigger the execution of ag__.if_stmt which performs the check
    @autograph
    def if_assign_partial(x: float):
        """Verify error from a conditional that doesn't produce a value in all branches."""

        if x < 3:
            y = 4

        return y

except AutoGraphError as e:
    # CHECK:   Some branches did not define a value for variable 'y'
    print(e)

# -----


# CHECK-LABEL: def if_assign_existing
@autograph
def if_assign_existing(x: float):
    """Test a conditional that assigns to an existing variable in all branches."""

    # CHECK:   y = 0
    y = 0

    # CHECK:   def if_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 4
    if x < 3:
        y = 4
    # CHECK:   def else_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 5
    else:
        y = 5

    # CHECK:   ag__.if_stmt(x < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   return y
    return y


print_code(if_assign_existing)

# -----


# CHECK-LABEL: def if_assign_existing_type_mismatch
@autograph
def if_assign_existing_type_mismatch(x: float):
    """Test a conditional that assigns to an existing variable with a different type, while being
    consistent across all branches."""

    # CHECK:   y = 0
    y = 0

    # CHECK:   def if_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 4.0
    if x < 3:
        y = 4.0
    # CHECK:   def else_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 5.0
    else:
        y = 5.0

    # CHECK:   ag__.if_stmt(x < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   return y
    return y


print_code(if_assign_existing_type_mismatch)

# -----


# CHECK-LABEL: def if_assign_existing_partial
@autograph
def if_assign_existing_partial(x: float):
    """Test a conditional that assigns to an existing variable in some branches only."""

    # CHECK:   y = 0
    y = 0

    # CHECK:   def if_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 4
    if x < 3:
        y = 4
    # CHECK:   def else_body():
    # CHECK:       nonlocal y
    # CHECK:       pass

    # CHECK:   ag__.if_stmt(x < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   return y
    return y


print_code(if_assign_existing_partial)

# -----

try:

    @qjit
    @autograph
    def if_assign_existing_partial_type_mismatch(x: float):
        """Verify error from a conditional that assigns to an existing value with different type,
        without defining a value in all branches. This should lead to a type mismatch error."""

        y = 0

        if x < 3:
            y = 4.0

        return y

except TypeError as e:
    # CHECK:   Conditional requires consistent return types across all branches
    print(e)


# -----


# CHECK-LABEL: def if_assign_multiple
@autograph
def if_assign_multiple(x: float):
    """Test a conditional that assigns to multiple existing variables."""

    # CHECK:   (y, z) = (0, False)
    y, z = 0, False

    # CHECK:       def if_body():
    # CHECK:           nonlocal {{[yz], [yz]}}
    # CHECK:           y = 4
    if x < 3:
        y = 4
    # CHECK:       def else_body():
    # CHECK:           nonlocal {{[yz], [yz]}}
    # CHECK-DAG:       y = 5
    # CHECK-DAG:       z = True
    else:
        y = 5
        z = True

    # CHECK:   ag__.if_stmt(x < 3, if_body, else_body, get_state, set_state, ('y', 'z'), 2)

    # CHECK:   return y * z
    return y * z


print_code(if_assign_multiple)

# -----


try:

    @qjit
    @autograph
    def if_assign_invalid_type(x: float):
        """Verify error from a conditional that produces a type invalid for tracing."""

        if x < 3:
            y = "hi"
        else:
            y = ""

        return len(y)

except TypeError as e:
    # CHECK:   Value 'hi' with type <class 'str'> is not a valid JAX type
    print(e)

# -----


# CHECK-LABEL: def if_elif
@autograph
def if_elif(x: float):
    """Test a conditional with more than two branches."""

    # CHECK:   y = 0
    y = 0

    # CHECK:   def if_body_1():
    # CHECK:       nonlocal y
    # CHECK:       y = 4
    if x < 3:
        y = 4
    # CHECK:   def else_body_1():
    # CHECK:       def if_body():
    # CHECK:           nonlocal y
    # CHECK:           y = 7
    # CHECK:       def else_body():
    # CHECK:           nonlocal y
    # CHECK:           pass
    # CHECK:       ag__.if_stmt(x < 5, if_body, else_body, get_state, set_state, ('y',), 1)
    elif x < 5:
        y = 7

    # CHECK:   ag__.if_stmt(x < 3, if_body_1, else_body_1, get_state_1, set_state_1, ('y',), 1)

    # CHECK:   return y
    return y


print_code(if_elif)

# -----


# CHECK-LABEL: def nested_call
def nested_call(x, y):
    """Nested function with conditional."""

    # CHECK:   def if_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 4
    if x < 3:
        y = 4
    # CHECK:   def else_body():
    # CHECK:       nonlocal y
    # CHECK:       pass

    # CHECK:   ag__.if_stmt(x < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   return y
    return y


# CHECK-LABEL: def if_call
@autograph
def if_call(x: float):
    """Test a conditional that is nested inside another function. All (user) functions invoked by
    the explicitly transformed function should also be transformed."""

    # CHECK:   y = 0
    y = 0

    # CHECK:   return ag__.converted_call(nested_call, (x, y)
    return nested_call(x, y)


if_call(0.1)  # needed to generate the source code for nested functions

print_code(nested_call)
print_code(if_call)
