# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the xDSL universe."""
from subprocess import run

import pytest
from xdsl.passes import ModulePass
from xdsl.universe import Universe as xUniverse

from catalyst.python_interface import dialects, transforms
from catalyst.python_interface.xdsl_universe import CATALYST_XDSL_UNIVERSE, shared_dialects

pytestmark = pytest.mark.xdsl

all_dialects = tuple(getattr(dialects, name) for name in dialects.__all__)
all_transforms = tuple(
    transform
    for name in transforms.__all__
    if isinstance((transform := getattr(transforms, name)), type)
    and issubclass(transform, ModulePass)
)


def test_correct_universe():
    """Test that all the available dialects and transforms are available in the universe."""
    for d in all_dialects:
        if d.name not in shared_dialects:
            assert d.name in CATALYST_XDSL_UNIVERSE.all_dialects
            assert CATALYST_XDSL_UNIVERSE.all_dialects[d.name]() == d

    for t in all_transforms:
        assert t.name in CATALYST_XDSL_UNIVERSE.all_passes
        assert CATALYST_XDSL_UNIVERSE.all_passes[t.name]() == t


def test_correct_multiverse():
    """Test that all the available dialects and transforms are available in the multiverse."""
    multiverse = xUniverse.get_multiverse()

    for d in all_dialects:
        assert d.name in multiverse.all_dialects
        if d.name not in shared_dialects:
            assert multiverse.all_dialects[d.name]() == d

    for t in all_transforms:
        assert t.name in multiverse.all_passes
        assert multiverse.all_passes[t.name]() == t


@pytest.mark.parametrize("_transform", all_transforms)
def test_xdsl_opt(_transform):
    """Test that Catalyst's dialects and transforms are available through xdsl-opt"""
    # quantum.device_init is needed for some passes to work correctly
    mod_string = """
        %shots = arith.constant 1 : i64
        %angle, %q0 = "test.op"() : () -> (f64, !quantum.bit)

        // For checking that the Quantum dialect is loaded correctly
        quantum.device shots(%shots) ["dummy_lib", "dummy.device", "{'dummy_kwarg': 0}"]

        // For checking that the Catalyst dialect is loaded correctly
        %list = catalyst.list_init : !catalyst.arraylist<i64>

        // For checking that the MBQC dialect is loaded correctly
        %mcm, %q1 = mbqc.measure_in_basis[XY, %angle] %q0 : i1, !quantum.bit

        // For checking that the PBC dialect is loaded correctly
        %q2 = pbc.fabricate zero : !quantum.bit
    """

    cmd = ["xdsl-opt", "-p", _transform.name]
    _ = run(cmd, input=mod_string, text=True, check=True)
