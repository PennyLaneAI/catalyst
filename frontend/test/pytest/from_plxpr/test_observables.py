# Copyright 2024 Xanadu Quantum Technologies Inc.

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
This module tests the from_plxpr conversion function.
"""

import jax
import numpy as np
import pennylane as qp
import pytest

from catalyst.from_plxpr import from_plxpr
from catalyst.from_plxpr.qref_jax_primitives import qref_get_p, qref_hermitian_p, qref_namedobs_p
from catalyst.jax_primitives import (
    expval_p,
    hamiltonian_p,
    tensorobs_p,
)

pytestmark = pytest.mark.usefixtures("disable_capture")


def test_hermitian():
    """Test a hermitian can be converted"""

    qp.capture.enable()

    @qp.qnode(qp.device("lightning.qubit", wires=2))
    def c(mat):
        return qp.expval(qp.Hermitian(mat, wires=(0, 1)))

    mat = (qp.X(0) @ qp.Y(1)).matrix()

    plxpr = jax.make_jaxpr(c)(mat)
    catalyst_xpr = from_plxpr(plxpr)(mat)

    qfunc = catalyst_xpr.eqns[0].params["call_jaxpr"]

    assert qfunc.eqns[4].primitive == qref_hermitian_p
    assert qfunc.eqns[4].params == {}
    assert qfunc.eqns[4].invars[0] == qfunc.invars[0]
    assert qfunc.eqns[4].invars[1] == qfunc.eqns[2].outvars[0]
    assert qfunc.eqns[4].invars[2] == qfunc.eqns[3].outvars[0]

    assert qfunc.eqns[5].invars[0] == qfunc.eqns[4].outvars[0]


def test_sprod():
    """Test that an sprod can be converted."""

    qp.capture.enable()

    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def c():
        return qp.expval(2 * qp.Z(0))

    jaxpr = jax.make_jaxpr(c)()
    catalyst_xpr = from_plxpr(jaxpr)()

    qfunc = catalyst_xpr.eqns[0].params["call_jaxpr"]
    assert qfunc.eqns[3].primitive == qref_namedobs_p
    assert qfunc.eqns[3].params == {"kind": "PauliZ"}

    # 4 is broadcast_in_dim
    assert qfunc.eqns[4].params["shape"] == (1,)

    assert qfunc.eqns[5].primitive == hamiltonian_p
    assert qfunc.eqns[5].invars[0] == qfunc.eqns[4].outvars[0]
    assert qfunc.eqns[5].invars[1] == qfunc.eqns[3].outvars[0]

    assert qfunc.eqns[6].primitive == expval_p
    assert qfunc.eqns[6].invars[0] == qfunc.eqns[5].outvars[0]


def test_prod():
    """Test the translation of a Prod"""

    qp.capture.enable()

    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def c():
        return qp.expval(qp.X(0) @ qp.Y(1) @ qp.Z(2))

    jaxpr = jax.make_jaxpr(c)()
    catalyst_xpr = from_plxpr(jaxpr)()
    qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]

    assert qfunc_xpr.eqns[2].primitive == qref_get_p
    assert qfunc_xpr.eqns[2].invars[1].val == 0

    assert qfunc_xpr.eqns[3].primitive == qref_namedobs_p
    assert qfunc_xpr.eqns[3].params == {"kind": "PauliX"}

    assert qfunc_xpr.eqns[4].primitive == qref_get_p
    assert qfunc_xpr.eqns[4].invars[1].val == 1

    assert qfunc_xpr.eqns[5].primitive == qref_namedobs_p
    assert qfunc_xpr.eqns[5].params == {"kind": "PauliY"}

    assert qfunc_xpr.eqns[6].primitive == qref_get_p
    assert qfunc_xpr.eqns[6].invars[1].val == 2

    assert qfunc_xpr.eqns[7].primitive == qref_namedobs_p
    assert qfunc_xpr.eqns[7].params == {"kind": "PauliZ"}

    assert qfunc_xpr.eqns[8].primitive == tensorobs_p
    assert qfunc_xpr.eqns[8].invars == [
        qfunc_xpr.eqns[3].outvars[0],
        qfunc_xpr.eqns[5].outvars[0],
        qfunc_xpr.eqns[7].outvars[0],
    ]


@pytest.mark.parametrize("as_linear_combination", (True, False))
def test_sum(as_linear_combination):
    """Test the conversion of a Sum class."""
    qp.capture.enable()

    @qp.qnode(qp.device("lightning.qubit", wires=4))
    def c():
        if as_linear_combination:
            # Note that we should investigate improving the capture of hamiltonian
            # so we don't have to unpack and repack the coefficients sc-95974
            H = qp.Hamiltonian(np.array([1, 2, 3]), [qp.X(0), qp.Y(1), qp.Z(2)])
        else:
            H = 1 * qp.X(0) + 2 * qp.Y(1) + 3 * qp.Z(2)
        return qp.expval(H)

    jaxpr = jax.make_jaxpr(c)()
    catalyst_xpr = from_plxpr(jaxpr)()
    qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]

    assert qfunc_xpr.eqns[2].primitive == qref_get_p
    assert qfunc_xpr.eqns[2].invars[1].val == 0

    assert qfunc_xpr.eqns[3].primitive == qref_namedobs_p
    assert qfunc_xpr.eqns[3].params == {"kind": "PauliX"}

    assert qfunc_xpr.eqns[4].primitive == qref_get_p
    assert qfunc_xpr.eqns[4].invars[1].val == 1

    assert qfunc_xpr.eqns[5].primitive == qref_namedobs_p
    assert qfunc_xpr.eqns[5].params == {"kind": "PauliY"}

    assert qfunc_xpr.eqns[6].primitive == qref_get_p
    assert qfunc_xpr.eqns[6].invars[1].val == 2

    assert qfunc_xpr.eqns[7].primitive == qref_namedobs_p
    assert qfunc_xpr.eqns[7].params == {"kind": "PauliZ"}

    # 8-11 broadcasting and concatenation
    assert qfunc_xpr.eqns[12].primitive == hamiltonian_p
    assert qfunc_xpr.eqns[12].invars[1:] == [
        qfunc_xpr.eqns[3].outvars[0],
        qfunc_xpr.eqns[5].outvars[0],
        qfunc_xpr.eqns[7].outvars[0],
    ]
