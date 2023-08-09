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

import inspect

from catalyst import qjit
from catalyst.autograph import AutographError, autograph


# CHECK-LABEL: if_simple(
@qjit
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


print(inspect.getsource(if_simple))

# -----


# CHECK-LABEL: if_else(
@qjit
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


print(inspect.getsource(if_else))

# -----


# CHECK-LABEL: if_assign(
@qjit
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


print(inspect.getsource(if_assign))

# -----


try:

    @qjit
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

    @qjit
    @autograph
    def if_assign_partial(x: float):
        if x < 3:
            y = 4

        return y

except AutographError as e:
    # CHECK:   Some branches did not define a value for variable 'y'
    print(e)

# -----


# CHECK-LABEL: if_assign_existing(
@qjit
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


print(inspect.getsource(if_assign_existing))

# -----


# CHECK-LABEL: if_assign_existing_type_mismatch(
@qjit
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


print(inspect.getsource(if_assign_existing_type_mismatch))

# -----


# CHECK-LABEL: if_assign_existing_partial(
@qjit
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


print(inspect.getsource(if_assign_existing_partial))

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


# CHECK-LABEL: if_assign_multiple(
@qjit
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


print(inspect.getsource(if_assign_multiple))

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


# CHECK-LABEL: if_elif(
@qjit
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


print(inspect.getsource(if_elif))
