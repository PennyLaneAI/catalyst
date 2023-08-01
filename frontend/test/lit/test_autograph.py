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

from catalyst.autograph import convert_cf


# CHECK-LABEL: if_simple(
@convert_cf
def if_simple(x: float):
    # CHECK:   @catalyst.cond(x < 3)
    # CHECK:   def {{.+}}():
    # CHECK:       pass
    if x < 3:
        pass

    return x


print(if_simple.__source__)

# -----


# CHECK-LABEL: if_else(
@convert_cf
def if_else(x: float):
    # CHECK:   @catalyst.cond(x < 3)
    # CHECK:   def [[fn:.+]]():
    # CHECK:       pass
    if x < 3:
        pass
    # CHECK:   @[[fn]].otherwise
    # CHECK:   def [[fn]]():
    # CHECK:       pass
    else:
        pass

    return x


print(if_else.__source__)

# -----


# CHECK-LABEL: if_assign(
@convert_cf
def if_assign(x: float):
    # CHECK:   @catalyst.cond(x < 3)
    # CHECK:   def [[fn:.+]]():
    # CHECK:       return 4
    if x < 3:
        y = 4
    # CHECK:   @[[fn]].otherwise
    # CHECK:   def [[fn]]():
    # CHECK:       return 5
    else:
        y = 5

    # CHECK:   y = [[fn]]()
    # CHECK:   return y
    return y


print(if_assign.__source__)

# -----


try:

    @convert_cf
    def if_assign_type_mismatch(x: float):
        if x < 3:
            y = 4.0
        else:
            y = 5

        return y

except Exception as e:
    # CHECK:   AutographException: Some branches did not have a consistent type for variable 'y'
    print(e)

# -----


try:

    @convert_cf
    def if_assign_partial(x: float):
        if x < 3:
            y = 4

        return y

except Exception as e:
    # CHECK:   AutographException: Some branches did not define a value for variable 'y'
    print(e)

# -----


# CHECK-LABEL: if_assign_existing(
@convert_cf
def if_assign_existing(x: float):
    # CHECK:   y = 0
    y = 0

    # CHECK:   @catalyst.cond(x < 3)
    # CHECK:   def [[fn:.+]]():
    # CHECK:       return 4
    if x < 3:
        y = 4
    # CHECK:   @[[fn]].otherwise
    # CHECK:   def [[fn]]():
    # CHECK:       return 5
    else:
        y = 5

    # CHECK:   y = [[fn]]()
    # CHECK:   return y
    return y


print(if_assign_existing.__source__)

# -----


# CHECK-LABEL: if_assign_existing_type_mismatch(
@convert_cf
def if_assign_existing_type_mismatch(x: float):
    # CHECK:   y = 0
    y = 0

    # CHECK:   @catalyst.cond(x < 3)
    # CHECK:   def [[fn:.+]]():
    # CHECK:       return 4.0
    if x < 3:
        y = 4.0
    # CHECK:   @[[fn]].otherwise
    # CHECK:   def [[fn]]():
    # CHECK:       return 5.0
    else:
        y = 5.0

    # CHECK:   y = [[fn]]()
    # CHECK:   return y
    return y


print(if_assign_existing_type_mismatch.__source__)

# -----


# CHECK-LABEL: if_assign_existing_partial(
@convert_cf
def if_assign_existing_partial(x: float):
    # CHECK:   y = 0
    y = 0

    # CHECK:   @catalyst.cond(x < 3)
    # CHECK:   def [[fn:.+]]():
    # CHECK:       return 4
    if x < 3:
        y = 4
    # CHECK:   @[[fn]].otherwise
    # CHECK:   def [[fn]]():
    # CHECK:       return y

    # CHECK:   y = [[fn]]()
    # CHECK:   return y
    return y


print(if_assign_existing_partial.__source__)

# -----

try:

    @convert_cf
    def if_assign_existing_partial_type_mismatch(x: float):
        y = 0

        if x < 3:
            y = 4.0

        return y

except Exception as e:
    # CHECK:   AutographException: Some branches did not have a consistent type for variable 'y'
    print(e)


# -----


# CHECK-LABEL: if_assign_multiple(
@convert_cf
def if_assign_multiple(x: float):
    # CHECK:   y, z = 0, False
    y, z = 0, False

    # CHECK:   @catalyst.cond(x < 3)
    # CHECK:   def [[fn:.+]]():
    # CHECK:       return 4, z
    if x < 3:
        y = 4
    # CHECK:   @[[fn]].otherwise
    # CHECK:   def [[fn]]():
    # CHECK:       return 5, True
    else:
        y = 5
        z = True

    # CHECK:   y, z = [[fn]]()
    # CHECK:   return y * z
    return y * z


print(if_assign_multiple.__source__)

# -----


try:

    @convert_cf
    def if_assign_invalid_type(x: float):
        if x < 3:
            y = "hi"
        else:
            y = ""

        return len(y)

except Exception as e:
    # CHECK:   AutographException: JIT-incompatible type encountered in if-clause assignment to variable 'y'
    print(e)

# -----


# CHECK-LABEL: if_elif(
@convert_cf
def if_elif(x: float):
    # CHECK:   y = 0
    y = 0

    # CHECK:   @catalyst.cond(x < 3)
    # CHECK:   def [[fn:.+]]():
    # CHECK:       return 4
    if x < 3:
        y = 4
    # CHECK:   @[[fn]].else_if
    # CHECK:   def [[fn]]():
    # CHECK:       return 7
    elif x < 5:
        y = 7
    # CHECK:   @[[fn]].otherwise
    # CHECK:   def [[fn]]():
    # CHECK:       return y

    # CHECK:   y = [[fn]]()
    # CHECK:   return y
    return y


print(if_elif.__source__)
