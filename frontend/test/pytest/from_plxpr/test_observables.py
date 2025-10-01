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
import pennylane as qml
import pytest

from catalyst.from_plxpr import from_plxpr
from catalyst.jax_primitives import (
    expval_p,
    hamiltonian_p,
    hermitian_p,
    namedobs_p,
    qextract_p,
    tensorobs_p,
)

pytestmark = pytest.mark.usefixtures("disable_capture")


def test_hermitian():
    """Test a hermitian can be converted"""

    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=2))
    def c(mat):
        return qml.expval(qml.Hermitian(mat, wires=(0, 1)))

    mat = (qml.X(0) @ qml.Y(1)).matrix()

    plxpr = jax.make_jaxpr(c)(mat)
    catalyst_xpr = from_plxpr(plxpr)(mat)

    qfunc = catalyst_xpr.eqns[0].params["call_jaxpr"]

    assert qfunc.eqns[4].primitive == hermitian_p
    assert qfunc.eqns[4].params == {}
    assert qfunc.eqns[4].invars[0] == qfunc.invars[0]
    assert qfunc.eqns[4].invars[1] == qfunc.eqns[2].outvars[0]
    assert qfunc.eqns[4].invars[2] == qfunc.eqns[3].outvars[0]

    assert qfunc.eqns[5].invars[0] == qfunc.eqns[4].outvars[0]


def test_sprod():
    """Test that an sprod can be converted."""

    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def c():
        return qml.expval(2 * qml.Z(0))

    jaxpr = jax.make_jaxpr(c)()
    catalyst_xpr = from_plxpr(jaxpr)()

    qfunc = catalyst_xpr.eqns[0].params["call_jaxpr"]
    assert qfunc.eqns[3].primitive == namedobs_p
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

    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def c():
        return qml.expval(qml.X(0) @ qml.Y(1) @ qml.Z(2))

    jaxpr = jax.make_jaxpr(c)()
    catalyst_xpr = from_plxpr(jaxpr)()
    qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]

    assert qfunc_xpr.eqns[2].primitive == qextract_p
    assert qfunc_xpr.eqns[2].invars[1].val == 0

    assert qfunc_xpr.eqns[3].primitive == namedobs_p
    assert qfunc_xpr.eqns[3].params == {"kind": "PauliX"}

    assert qfunc_xpr.eqns[4].primitive == qextract_p
    assert qfunc_xpr.eqns[4].invars[1].val == 1

    assert qfunc_xpr.eqns[5].primitive == namedobs_p
    assert qfunc_xpr.eqns[5].params == {"kind": "PauliY"}

    assert qfunc_xpr.eqns[6].primitive == qextract_p
    assert qfunc_xpr.eqns[6].invars[1].val == 2

    assert qfunc_xpr.eqns[7].primitive == namedobs_p
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
    qml.capture.enable()

    @qml.qnode(qml.device("lightning.qubit", wires=4))
    def c():
        if as_linear_combination:
            # Note that we should investigate improving the capture of hamiltonian
            # so we don't have to unpack and repack the coefficients sc-95974
            H = qml.Hamiltonian(np.array([1, 2, 3]), [qml.X(0), qml.Y(1), qml.Z(2)])
        else:
            H = 1 * qml.X(0) + 2 * qml.Y(1) + 3 * qml.Z(2)
        return qml.expval(H)

    jaxpr = jax.make_jaxpr(c)()
    catalyst_xpr = from_plxpr(jaxpr)()
    qfunc_xpr = catalyst_xpr.eqns[0].params["call_jaxpr"]

    assert qfunc_xpr.eqns[2].primitive == qextract_p
    assert qfunc_xpr.eqns[2].invars[1].val == 0

    assert qfunc_xpr.eqns[3].primitive == namedobs_p
    assert qfunc_xpr.eqns[3].params == {"kind": "PauliX"}

    assert qfunc_xpr.eqns[4].primitive == qextract_p
    assert qfunc_xpr.eqns[4].invars[1].val == 1

    assert qfunc_xpr.eqns[5].primitive == namedobs_p
    assert qfunc_xpr.eqns[5].params == {"kind": "PauliY"}

    assert qfunc_xpr.eqns[6].primitive == qextract_p
    assert qfunc_xpr.eqns[6].invars[1].val == 2

    assert qfunc_xpr.eqns[7].primitive == namedobs_p
    assert qfunc_xpr.eqns[7].params == {"kind": "PauliZ"}

    # 8-11 broadcasting and concatenation
    assert qfunc_xpr.eqns[12].primitive == hamiltonian_p
    assert qfunc_xpr.eqns[12].invars[1:] == [
        qfunc_xpr.eqns[3].outvars[0],
        qfunc_xpr.eqns[5].outvars[0],
        qfunc_xpr.eqns[7].outvars[0],
    ]
