# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MLIR Dialect for Qref dialect."""

# pylint: disable=relative-beyond-top-level
from ._qref_ops_gen import *  # noqa: F401
from ._qref_ops_gen import OperatorOp as _OperatorOpGen
from ._ods_common import (
    get_default_loc_context as _ods_get_default_loc_context,
    get_op_results_or_values as _get_op_results_or_values,
)
from ._ods_common import _cext as _ods_cext
_ods_ir = _ods_cext.ir


class OperatorOp(_OperatorOpGen):
    def __init__(self, op_name, params, forward_args, qubits, ctrl_qubits,
                 ctrl_values, arr_qubit_indices=None, adjoint=False, UID=None, *,
                 qreg=None, arr_ctrl_indices=None, arr_ctrl_values=None,
                 static_data=None, param_map=None, qubit_map=None, loc=None, ip=None):
        operands = []
        attributes = {}
        # op_name is a StringProp — must go in attributes
        attributes["op_name"] = (op_name if isinstance(op_name, _ods_ir.Attribute)
                                  else _ods_ir.StringAttr.get(op_name))
        operands.append(_get_op_results_or_values(params))
        operands.append(_get_op_results_or_values(forward_args))
        operands.append(_get_op_results_or_values(qubits))
        operands.append(_get_op_results_or_values(ctrl_qubits))
        operands.append(ctrl_values if ctrl_values is not None else [])
        operands.append(_get_op_results_or_values(qreg))
        operands.append(arr_qubit_indices if arr_qubit_indices is not None else [])
        operands.append(arr_ctrl_indices)
        operands.append(arr_ctrl_values) 
        _ods_context = _ods_get_default_loc_context(loc)
        if bool(adjoint):
            attributes["adjoint"] = _ods_ir.UnitAttr.get(_ods_context)
        if UID is not None:
            attributes["UID"] = _ods_ir.IntegerAttr.get(
                _ods_ir.IntegerType.get_signless(64), UID)
        if static_data is not None:
            attributes["static_data"] = static_data
        if param_map is not None:
            attributes["param_map"] = param_map
        if qubit_map is not None:
            attributes["qubit_map"] = qubit_map
        # bypass _OperatorOpGen.__init__ and go directly to OpView
        super(_OperatorOpGen, self).__init__(
            self.OPERATION_NAME, self._ODS_REGIONS, self._ODS_OPERAND_SEGMENTS,
            self._ODS_RESULT_SEGMENTS, attributes=attributes, results=[],
            operands=operands, successors=None, regions=None, loc=loc, ip=ip)