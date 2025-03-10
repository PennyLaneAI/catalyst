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

from catalyst import (
    AutoGraphError,
    autograph_source,
    disable_autograph,
    qjit,
    run_autograph,
)
from catalyst.utils.dummy import dummy_func


def print_code(fn):
    """Print autograph generated code for a function."""
    print(autograph_source(fn))


# CHECK-LABEL: def ag__while_simple
@run_autograph
def while_simple(x: float):
    """Test a simple while-loop statement."""

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
    # CHECK: while
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


# CHECK-LABEL: def ag__if_simple
@run_autograph
def if_simple(x: float):
    """Test a simple conditional with a single branch."""

    # CHECK:   def if_body():
    # CHECK:       pass
    if x < 3:
        pass
    # CHECK:   def else_body():
    # CHECK:       pass

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body, else_body, get_state, set_state, (), 0)

    return x


print_code(if_simple)

# -----


# CHECK-LABEL: def ag__if_else
@run_autograph
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

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body, else_body, get_state, set_state, (), 0)

    return x


print_code(if_else)

# -----


# CHECK-LABEL: def ag__if_assign
@run_autograph
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

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.ld(y)
    # CHECK:   return fscope.ret(retval_, do_return)
    return y


print_code(if_assign)

# -----


# CHECK-LABEL: def ag__if_assign_no_type_mismatch
@qjit  # needed to trigger Catalyst type checks during tracing
@run_autograph
def if_assign_no_type_mismatch(x: float):
    """Verify the absense of error from a conditional that doesn't produce the same type across
    branches."""

    # CHECK:   def if_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 4.0
    if x < 3:
        y = 4.0
    # CHECK:   def else_body():
    # CHECK:       nonlocal y
    # CHECK:       y = 5
    else:
        y = 5

    return y


print_code(if_assign_no_type_mismatch)

# -----


try:

    @qjit  # needed to trigger the execution of ag__.if_stmt which performs the check
    @run_autograph
    def if_assign_pytree_shape_mismatch(x: float):
        """Verify error from a conditional that doesn't produce a value in all branches."""

        if x < 3:
            y = (4, 4)
        else:
            y = 4

        return y

except TypeError as e:
    # CHECK:   Conditional requires a consistent return structure across all branches
    print(e)

# -----


try:

    @qjit  # needed to trigger the execution of ag__.if_stmt which performs the check
    @run_autograph
    def if_assign_partial(x: float):
        """Verify error from a conditional that doesn't produce a value in all branches."""

        if x < 3:
            y = 4

        return y

except AutoGraphError as e:
    # CHECK:   Some branches did not define a value for variable 'y'
    print(e)

# -----


# CHECK-LABEL: def ag__if_assign_existing
@run_autograph
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

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.ld(y)
    # CHECK:   return fscope.ret(retval_, do_return)
    return y


print_code(if_assign_existing)

# -----


# CHECK-LABEL: def ag__if_assign_existing_type_mismatch
@run_autograph
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

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.ld(y)
    # CHECK:   return fscope.ret(retval_, do_return)
    return y


print_code(if_assign_existing_type_mismatch)

# -----


# CHECK-LABEL: def ag__if_assign_existing_partial
@run_autograph
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

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.ld(y)
    # CHECK:   return fscope.ret(retval_, do_return)
    return y


print_code(if_assign_existing_partial)

# -----


# CHECK-LABEL: def ag__if_assign_existing_partial_no_type_mismatch
@qjit
@run_autograph
def if_assign_existing_partial_no_type_mismatch(x: float):
    """Verify error from a conditional that assigns to an existing value with different type,
    without defining a value in all branches. This should lead to a type mismatch error."""

    # CHECK: y = 0
    y = 0

    # CHECK:       def if_body():
    # CHECK:           nonlocal y
    # CHECK:           y = 4.0
    if x < 3:
        y = 4.0

    # CHECK:       def else_body():
    # CHECK:           nonlocal y
    # CHECK:           pass

    return y


print_code(if_assign_existing_partial_no_type_mismatch)

# -----


# CHECK-LABEL: def ag__if_assign_multiple
@run_autograph
def if_assign_multiple(x: float):
    """Test a conditional that assigns to multiple existing variables."""

    # CHECK:   {{\(?}}y, z{{\)?}} = (0, False)
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

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body, else_body, get_state, set_state, ('y', 'z'), 2)

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.ld(y) * ag__.ld(z)
    # CHECK:   return fscope.ret(retval_, do_return)
    return y * z


print_code(if_assign_multiple)

# -----


try:

    @qjit
    @run_autograph
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


# CHECK-LABEL: def ag__if_elif
@run_autograph
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
    # CHECK:       ag__.if_stmt(ag__.ld(x) < 5, if_body, else_body, get_state, set_state, ('y',), 1)
    elif x < 5:
        y = 7

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body_1, else_body_1, get_state_1, set_state_1, ('y',), 1)

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.ld(y)
    # CHECK:   return fscope.ret(retval_, do_return)
    return y


print_code(if_elif)

# -----


# CHECK-LABEL: def ag__nested_call
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

    # CHECK:   ag__.if_stmt(ag__.ld(x) < 3, if_body, else_body, get_state, set_state, ('y',), 1)

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.ld(y)
    # CHECK:   return fscope.ret(retval_, do_return)
    return y


# CHECK-LABEL: def ag__if_call
@run_autograph
def if_call(x: float):
    """Test a conditional that is nested inside another function. All (user) functions invoked by
    the explicitly transformed function should also be transformed."""

    # CHECK:   y = 0
    y = 0

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.converted_call(ag__.ld(nested_call), (ag__.ld(x), ag__.ld(y))
    # CHECK:   return fscope.ret(retval_, do_return)
    return nested_call(x, y)


if_call(0.1)  # needed to generate the source code for nested functions

print_code(nested_call)
print_code(if_call)

# -----


# CHECK-LABEL: def ag__logical_calls
@run_autograph
def logical_calls(x: float, y: float):
    """Check that catalyst can handle ``and``, ``or`` and ``not`` using autograph."""
    # pylint: disable=chained-comparison

    # CHECK:   a = ag__.and_
    a = x >= 0.0 and x <= 1.0

    # CHECK:   b = ag__.not_
    b = not y >= 1.0

    # CHECK:   try:
    # CHECK:     do_return = True
    # CHECK:     retval_ = ag__.or_(lambda{{\ ?}}: ag__.ld(a), lambda{{\ ?}}: ag__.ld(b))
    # CHECK:   return fscope.ret(retval_, do_return)
    return a or b


print_code(logical_calls)


# -----


# CHECK-LABEL: def ag__chain_logical_call
@run_autograph
def chain_logical_call(x: float):
    """Check that catalyst can handle chained-``and`` using autograph."""

    # CHECK:        ag__.and_
    # CHECK-SAME:     0.0 <= ag__.ld(x)
    # CHECK-SAME:     ag__.ld(x) <= 1.0
    return 0.0 <= x <= 1.0


print_code(chain_logical_call)


# -----


@disable_autograph
def f():
    """Simple function with if statements"""
    x = 6
    if x > 5:
        y = x**2
    else:
        y = x**3
    return y


# CHECK-LABEL: def disable_autograph_decorator_jax
@qjit(autograph=True)
def disable_autograph_decorator_jax(x: float, n: int):
    """Checks that Autograph is disabled for a given function."""
    # CHECK: body_jaxpr={ lambda ; d:i64[] e:f64[]. let f:f64[] = add e 36.0 in (f,) }
    for _ in range(n):
        x = x + f()
    return x


print("def disable_autograph_decorator_jax")
print(disable_autograph_decorator_jax.jaxpr)


# -----


def g():
    """Simple function with if statements"""
    x = 6
    if x > 5:
        y = x**2
    else:
        y = x**3
    return y


# CHECK-LABEL: def enable_autograph_decorator_jax
@qjit(autograph=True)
def enable_autograph_decorator_jax(x: float, n: int):
    """Checks that Autograph is enabled for a given function."""
    # CHECK: branch_jaxprs=[{ lambda ; . let  in (36,) }, { lambda ; . let  in (216,) }]
    for _ in range(n):
        x = x + g()
    return x


print("def enable_autograph_decorator_jax")
print(enable_autograph_decorator_jax.jaxpr)


# -----


def h():
    """Simple function with if statements"""
    x = 6
    if x > 5:
        y = x**2
    else:
        y = x**3
    return y


# CHECK-LABEL: def disable_autograph_context_manager_jax
@qjit(autograph=True)
def disable_autograph_context_manager_jax():
    """Checks that Autograph is disabled for a given context."""
    # CHECK: { lambda ; . let in (36.4,) }
    x = 0.4
    with disable_autograph:
        x += h()
    return x


print("def disable_autograph_context_manager_jax")
print(disable_autograph_context_manager_jax.jaxpr)


# -----


def func():
    """Simple function with if statements"""
    x = 6
    if x > 5:
        y = x**2
    else:
        y = x**3
    return y


# CHECK-LABEL: def enable_autograph_context_manager_jax
@qjit(autograph=True)
def enable_autograph_context_manager_jax():
    """Checks that Autograph is enabled with no context."""
    # CHECK: branch_jaxprs=[{ lambda ; . let  in (36,) }, { lambda ; . let  in (216,) }]
    x = 0.4
    x += func()
    return x


print("def enable_autograph_context_manager_jax")
print(enable_autograph_context_manager_jax.jaxpr)


# -----


# CHECK-LABEL: def include_module_to_autograph
@qjit(autograph=True, autograph_include=["catalyst.utils.dummy"])
def include_module_to_autograph(x: float, n: int):
    """Checks that a module is included to Autograph conversion."""
    # CHECK: branch_jaxprs=[{ lambda ; . let  in (36,) }, { lambda ; . let  in (216,) }]
    for _ in range(n):
        x = x + dummy_func(6)
    return x


print("def include_module_to_autograph")
print(include_module_to_autograph.jaxpr)


# -----


# CHECK-LABEL: def excluded_module_from_autograph
@qjit(autograph=True)
def excluded_module_from_autograph(x: float, n: int):
    """Checks that a module is excluded from Autograph conversion."""
    # CHECK: body_jaxpr={ lambda ; d:i64[] e:f64[]. let f:f64[] = add e 36.0 in (f,) }
    for _ in range(n):
        x = x + dummy_func(6)
    return x


print("def excluded_module_from_autograph")
print(excluded_module_from_autograph.jaxpr)
