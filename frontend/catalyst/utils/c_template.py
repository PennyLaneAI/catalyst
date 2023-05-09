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

"""This module contains templating functionality to create a C program
that calls the generated function. This is mostly useful for debugging
as sometimes it is better to have a Python-less exection environment.
"""

import numpy as np
from jax._src.lib.mlir import ir


def mlir_type_to_numpy_type(t):
    """Convert an MLIR type to a Numpy type.

    Args:
        t: an MLIR numeric type
    Returns:
        A numpy type
    Raises:
        TypeError
    """
    retval = None
    if ir.ComplexType.isinstance(t):
        base = ir.ComplexType(t).element_type
        if ir.F64Type.isinstance(base):
            retval = np.complex128
        else:
            retval = np.complex64
    elif ir.F64Type.isinstance(t):
        retval = np.float64
    elif ir.F32Type.isinstance(t):
        retval = np.float32
    elif ir.IntegerType.isinstance(t):
        int_t = ir.IntegerType(t)
        if int_t.width == 1:
            retval = np.bool_
        elif int_t.width == 8:
            retval = np.int8
        elif int_t.width == 16:
            retval = np.int16
        elif int_t.width == 32:
            retval = np.int32
        else:
            retval = np.int64

    if retval is None:
        raise TypeError("Requested return type is unavailable.")
    return retval


class Type:
    """Interface for types as needed by this template."""

    def __init__(self, name, decl):
        self._name = name
        self._decl = decl

    @property
    def name(self):
        """Type's name"""
        return self._name

    @property
    def decl(self):
        """Type's declaration"""
        return self._decl

    def __eq__(self, other):
        return hash(self.name) == hash(other.name)

    def __hash__(self):
        return hash(self.name)


class Variable:
    """Interface for variables as needed by this template."""

    def __init__(self, name, typ, init):
        self._name = name
        self._typ = typ
        self._init = init

    @property
    def name(self):
        """Variable's name"""
        return self._name

    @property
    def typ(self):
        """Variable's type"""
        return self._typ

    @property
    def init(self):
        """Variable's initialization"""
        return self._init


class ResultVar(Variable):
    """Template for handling the result type and result variable."""

    def __init__(self, resultType):
        element_types = [ir.RankedTensorType(y).element_type for y in resultType]
        numpyTypes = [mlir_type_to_numpy_type(y) for y in element_types]
        dependencies = []
        for dtype, mlir_type in zip(numpyTypes, resultType):
            typ = np.dtype(dtype)
            rank = len(ir.RankedTensorType(mlir_type).shape)
            dependencies.append(CType(typ, rank))

        type_decl = ResultVar._result_type_definition(dependencies)
        self._dependencies = set(typ for typ in dependencies)

        name = ResultVar._get_variable_name()
        typ = Type(ResultVar._get_type_name(), type_decl)
        init = ResultVar._get_variable_instantiation()

        super().__init__(name, typ, init)

    @property
    def dependencies(self):
        """Type dependencies. Since the result variable is a struct,
        each field may be a different type. We must ensure that these types
        are defined before the struct itself is defined. Here we provide a set
        of field types. They must be a set as that ensure that there are no multiple
        definitions of the same field type."""
        return self._dependencies

    @staticmethod
    def _result_type_definition(field_types):
        fields = []
        for idx, field_t in enumerate(field_types):
            fields.append(f"{field_t.name} f_{idx};")
        fields = "\n\t".join(fields)
        result_t = ResultVar._get_type_name()
        return f"""\n{result_t} {{
\t{fields}
}};
        """

    @staticmethod
    def _get_variable_instantiation():
        typ = ResultVar._get_type_name()
        name = ResultVar._get_variable_name()
        return f"{typ} {name};"

    @staticmethod
    def _get_type_name():
        return "struct result_t"

    @staticmethod
    def _get_variable_name():
        return "result_val"


class CType(Type):
    """A class that holds necessary information for templating a C Struct type.

    This class will be printed to a memref type.
    """

    def __init__(self, typ, rank):
        name = CType._get_name(typ, rank)
        decl = CType._get_definition(name, typ, rank)
        super().__init__(name, decl)

    @staticmethod
    def _get_template_for_sizes_and_strides(rank):
        rank_0 = ""
        rank_n_bt_0 = f"""size_t sizes[{rank}];
 \tsize_t strides[{rank}];"""
        return rank_n_bt_0 if rank else rank_0

    @staticmethod
    def _get_name(typ, rank):
        return f"struct memref_{typ}x{rank}_t"

    @staticmethod
    def _get_definition(name, typ, rank):
        sizes_and_strides = CType._get_template_for_sizes_and_strides(rank)
        return f"""
{name}
{{
\t{typ}* allocated;
\t{typ}* aligned;
\tsize_t offset;
\t{sizes_and_strides}
}};"""


class CVariable(Variable):
    """Memref variables used in the C program."""

    def __init__(self, array, arg_idx):
        rank = len(array.shape)
        typ = CType(array.dtype, rank)
        name = CVariable._get_variable_name(arg_idx)
        init = CVariable._get_initialization(array, arg_idx)
        super().__init__(name, typ, init)

    @staticmethod
    def _get_buffer_name(idx):
        return f"buff_{idx}"

    @staticmethod
    def _get_variable_name(idx):
        return f"arg_{idx}"

    @staticmethod
    def _get_buffer_size(array):
        rank = len(array.shape)
        return len(array.data.tolist()) if rank > 0 else 0

    @staticmethod
    def _get_initialization(array, idx):
        rank = len(array.shape)
        typ = CType(array.dtype, rank)
        buff_name = CVariable._get_buffer_name(idx)
        var_name = CVariable._get_variable_name(idx)
        sizes = CVariable._get_sizes(array)
        strides = CVariable._get_strides(array)
        buff_size = CVariable._get_buffer_size(array)
        elements = CVariable._get_array_data(array)
        fmt_rank_bt_0 = f"""
\t{array.dtype} {buff_name}[{buff_size}] = {{ { elements } }};
\t{typ.name} {var_name} = {{ {buff_name}, {buff_name}, 0, {{ {sizes} }}, {{ {strides} }}, }};
        """
        fmt_rank_0 = f"""
\t{array.dtype} {buff_name} = { elements };
\t{typ.name} {var_name} = {{ &{buff_name}, &{buff_name}, 0 }};
        """
        return fmt_rank_bt_0 if rank > 0 else fmt_rank_0

    @staticmethod
    def _get_array_data(array):
        rank = len(array.shape)
        if rank == 0:
            return array
        elements = [str(element) for element in array.data.tolist()]
        elements_str = ",".join(elements)
        return elements_str

    @staticmethod
    def _get_sizes(array):
        sizes = [str(dim) for dim in array.shape]
        sizes_str = ",".join(sizes)
        return sizes_str

    @staticmethod
    def _get_strides(array):
        strides = [str(stride // 8) for stride in array.strides]
        strides_str = ",".join(strides)
        return strides_str


def get_template(func_name, restype, *args):
    """Get a C program template for the current function
    with the arguments specified by *args.
    """
    cvars = []
    for idx, arg in enumerate(args):
        cvars.append(CVariable(arg, idx))

    variables = cvars
    types = {}
    if restype:
        result = ResultVar(restype)
        variables = [result] + variables
        types = set(typ for typ in result.dependencies)

    variables_initialization = "".join(var.init for var in variables)
    arg_vars = ", ".join("&" + var.name for var in variables)
    arg_types = ", ".join(var.typ.name + "*" for var in variables)

    types.update(set(var.typ for var in cvars))
    types = list(types)

    if restype:
        # The result type must be at the end,
        # Otherwise a field in the result structure
        # might be defined afterwards resulting in a warning/error.
        types += [result.typ]

    types = "".join(typ.decl for typ in types)

    template = f"""
#include <complex.h>
#include <stddef.h>
#include <stdint.h>

typedef int64_t int64;
typedef double float64;
typedef float float32;
typedef double complex complex128;
typedef float complex complex64;

{types}

extern void setup(int, char**);
extern void _catalyst_ciface_{func_name}({arg_types});
extern void teardown();

int
main(int argc, char** argv)
{{

\t{variables_initialization}

\tsetup(1, &argv[0]);
\t_catalyst_ciface_{func_name}({arg_vars});
\tteardown();
}}
    """
    return template
