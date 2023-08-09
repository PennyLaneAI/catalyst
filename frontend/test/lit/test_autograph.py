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

# RUN: %PYTHON %s | FileCheck %s

from catalyst import qjit
from catalyst.autograph import AutoGraphError, autograph, print_code


# CHECK-LABEL: def if_simple
@autograph
def if_simple(x: float):
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
    # CHECK:   y = 0
    y = 0

    # CHECK:   return ag__.converted_call(nested_call, (x, y)
    return nested_call(x, y)

if_call(0.1)  # needed to generate the source code for nested functions

print_code(nested_call)
print_code(if_call)
