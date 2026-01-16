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
"""This file contains a script for comparing Catalyst's MLIR dialects
with their xDSL versions."""

from argparse import ArgumentParser
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from re import fullmatch
from sys import modules as sys_modules
from types import ModuleType

from xdsl.dialects.builtin import UnitAttr
from xdsl.ir import Attribute, Dialect, Operation
from xdsl.irdl import (
    AttributeDef,
    AttrOrPropDef,
    BaseAttr,
    OpDef,
    OperandDef,
    OptionalDef,
    PropertyDef,
    RegionDef,
    ResultDef,
)

from catalyst.python_interface import dialects as xdialects

all_dialects = {
    "catalyst": "Catalyst",
    "mbqc": "MBQC",
    "qec": "QEC",
    "quantum": "Quantum",
}
"""Dictionary mapping the macro corresponding to a dialect its name."""

ignored_keys = ("operandSegmentSizes", "resultSegmentSizes")


def import_from_path(module_name, file_path) -> ModuleType:
    """Dynamically import a source file/folder as a module.
    Reference: https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = spec_from_file_location(module_name, file_path)
    module = module_from_spec(spec)
    sys_modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_dialect(catalyst_dialect: Dialect, gen_dialect: Dialect) -> None:
    """Compare xDSL dialects against their TableGen definitions.

    Args:
        catalyst_dialect (xdsl.ir.Dialect): Catalyst dialect to verify
        gen_dialect (xdsl.ir.Dialect): generated dialect for comparison

    Raises:
        RuntimeError: if verification failed
    """
    assert catalyst_dialect.name == gen_dialect.name, "The name of the dialect does not match."

    c_attrs: list[Attribute] = sorted(
        list(catalyst_dialect.attributes), key=lambda attr: attr.__name__
    )
    g_attrs: list[Attribute] = sorted(list(gen_dialect.attributes), key=lambda attr: attr.__name__)
    assert len(c_attrs) == len(g_attrs), "The number of attributes does not match."

    c_ops: list[Operation] = sorted(list(catalyst_dialect.operations), key=lambda op: op.__name__)
    g_ops: list[Operation] = sorted(list(gen_dialect.operations), key=lambda op: op.__name__)
    assert len(c_ops) == len(g_ops), "The number of operations in the dialect does not match."

    # Check that both dialects have the same attributes
    c_attr_names: set[str] = set(attr.__name__ for attr in c_attrs)
    g_attr_names: set[str] = set(attr.__name__ for attr in g_attrs)
    assert c_attr_names == g_attr_names, (
        "Mismatch between xDSL and MLIR dialect attributes.\n"
        "Attributes in the MLIR dialect that are not in the xDSL dialect: "
        f"{g_attr_names - c_attr_names}\n"
        "Attributes in the xDSL dialect that are not in the MLIR dialect: "
        f"{c_attr_names - g_attr_names}"
    )

    # Check that both dialects have the same operations
    c_op_names: set[str] = set(op.__name__ for op in c_ops)
    g_op_names: set[str] = set(op.__name__ for op in g_ops)
    assert c_op_names == g_op_names, (
        "Mismatch between xDSL and MLIR dialect operations.\n"
        "Operations in the MLIR dialect that are not in the xDSL dialect: "
        f"{g_op_names - c_op_names}\n"
        "Operations in the xDSL dialect that are not in the MLIR dialect: "
        f"{c_op_names - g_op_names}"
    )

    # Since we sorted the attributes, now that we know both dialects have the
    # same attributes, using zip this way should work correctly
    for c_attr, g_attr in zip(c_attrs, g_attrs):
        assert c_attr.name == g_attr.name, (
            f"The 'name' field of '{c_attr.__name__}' does not match. "
            f"xDSL: '{c_attr.name}', MLIR: '{g_attr.name}'"
        )

    # Since we sorted the operations, now that we know both dialects have the
    # same ops, using zip this way should work correctly
    for c_op, g_op in zip(c_ops, g_ops):
        assert c_op.name == g_op.name, (
            f"The 'name' field of '{c_op.__name__}' does not match. "
            f"xDSL: '{c_op.name}', MLIR: '{g_op.name}'"
        )

        c_def: OpDef = c_op.get_irdl_definition()
        g_def: OpDef = g_op.get_irdl_definition()

        verify_operands(c_def, g_def)
        verify_properties(c_def, g_def)
        verify_attributes(c_def, g_def)
        verify_regions(c_def, g_def)
        verify_results(c_def, g_def)


def verify_operands(c_def: OpDef, g_def: OpDef) -> None:
    """Verify operation operands"""
    assert len(c_def.operands) == len(
        g_def.operands
    ), f"The number of operands for {c_def.name} does not match."

    c_operands: dict[str, OperandDef] = dict(c_def.operands)
    c_operand_names: set[str] = set(c_operands.keys())
    g_operands: dict[str, OperandDef] = dict(g_def.operands)
    g_operand_names: set[str] = set(g_operands.keys())
    assert c_operand_names == g_operand_names, (
        f"Mismatch between operand names for {c_def.name}.\n"
        f"xDSL operands: {c_operand_names}, MLIR operands: {g_operand_names}"
    )

    for operand_name in c_operand_names:
        # Verifying that optional, variadic, and default operands match
        assert type(c_operands[operand_name]) == type(g_operands[operand_name]), (
            f"Mismatch between types for '{operand_name}' operand of {c_def.name}. "
            f"xDSL operand type: {type(c_operands[operand_name])}, "
            f"MLIR operand type: {type(g_operands[operand_name])}"
        )


def verify_properties(c_def: OpDef, g_def: OpDef) -> None:
    """Verify operation properties"""
    c_props: dict[str, PropertyDef] = c_def.properties
    g_props: dict[str, PropertyDef] = g_def.properties
    _ = [c_props.pop(ignored_key, None) for ignored_key in ignored_keys]
    _ = [g_props.pop(ignored_key, None) for ignored_key in ignored_keys]

    assert len(c_props) == len(
        g_props
    ), f"The number of properties for {c_def.name} does not match."

    c_prop_names: set[str] = set(c_props.keys())
    g_prop_names: set[str] = set(g_props.keys())
    assert c_prop_names == g_prop_names, (
        f"Mismatch between property names for {c_def.name}.\n"
        f"xDSL properties: {c_prop_names}, MLIR properties: {g_prop_names}"
    )

    for prop_name in c_prop_names:
        # Verifying that optional, variadic, and default properties match
        assert type(c_props[prop_name]) == type(g_props[prop_name]), (
            f"Mismatch between types for '{prop_name}' property of {c_def.name}. "
            f"xDSL property type: {type(c_props[prop_name])}, "
            f"MLIR property type: {type(g_props[prop_name])}"
        )
        assert_unit_attr(c_props[prop_name], g_props[prop_name], c_def.name, prop_name)


def verify_attributes(c_def: OpDef, g_def: OpDef) -> None:
    """Verify operation attributes"""
    c_attrs: dict[str, AttributeDef] = c_def.attributes
    g_attrs: dict[str, AttributeDef] = g_def.attributes
    _ = [c_attrs.pop(ignored_key, None) for ignored_key in ignored_keys]
    _ = [g_attrs.pop(ignored_key, None) for ignored_key in ignored_keys]

    assert len(c_attrs) == len(
        g_attrs
    ), f"The number of attributes for {c_def.name} does not match."

    c_attr_names: set[str] = set(c_attrs.keys())
    g_attr_names: set[str] = set(g_attrs.keys())
    assert c_attr_names == g_attr_names, (
        f"Mismatch between attribute names for {c_def.name}.\n"
        f"xDSL attributes: {c_attr_names}, MLIR attributes: {g_attr_names}"
    )

    for attr_name in c_attr_names:
        # Verifying that optional, variadic, and default attributes match
        assert type(c_attrs[attr_name]) == type(g_attrs[attr_name]), (
            f"Mismatch between types for '{attr_name}' attribute of {c_def.name}. "
            f"xDSL attribute type: {type(c_attrs[attr_name])}, "
            f"MLIR attribute type: {type(g_attrs[attr_name])}"
        )
        assert_unit_attr(c_attrs[attr_name], g_attrs[attr_name], c_def.name, attr_name)


def verify_regions(c_def: OpDef, g_def: OpDef) -> None:
    """Verify operation regions"""
    assert len(c_def.regions) == len(
        g_def.regions
    ), f"The number of regions for {c_def.name} does not match."

    c_regions: dict[str, RegionDef] = dict(c_def.regions)
    c_region_names: set[str] = set(c_regions.keys())
    g_regions: dict[str, RegionDef] = dict(g_def.regions)
    g_region_names: set[str] = set(g_regions.keys())
    assert c_region_names == g_region_names, (
        f"Mismatch between region names for {c_def.name}.\n"
        f"xDSL regions: {c_region_names}, MLIR regions: {g_region_names}"
    )

    for region_name in c_region_names:
        # Verifying that optional, variadic, and default regions match
        assert type(c_regions[region_name]) == type(g_regions[region_name]), (
            f"Mismatch between types for '{region_name}' region of {c_def.name}. "
            f"xDSL region type: {type(c_regions[region_name])}, "
            f"MLIR region type: {type(g_regions[region_name])}"
        )


def verify_results(c_def: OpDef, g_def: OpDef) -> None:
    """Verify operation results"""
    assert len(c_def.results) == len(
        g_def.results
    ), f"The number of results for {c_def.name} does not match."

    c_results: dict[str, ResultDef] = dict(c_def.results)
    c_result_names: set[str] = set(c_results.keys())
    g_results: dict[str, ResultDef] = dict(g_def.results)
    g_result_names: set[str] = set(g_results.keys())

    if any(fullmatch(r"v[0-9]+", result_name) for result_name in g_result_names):
        # If there are any unnamed results, xdsl-tblgen will rename them to
        # vX, where X is an integer
        return

    assert c_result_names == g_result_names, (
        f"Mismatch between result names for {c_def.name}.\n"
        f"xDSL results: {c_result_names}, MLIR results: {g_result_names}"
    )

    for result_name in c_result_names:
        # Verifying that optional, variadic, and default results match
        assert type(c_results[result_name]) == type(g_results[result_name]), (
            f"Mismatch between types for '{result_name}' result of {c_def.name}. "
            f"xDSL result type: {type(c_results[result_name])}, "
            f"MLIR result type: {type(g_results[result_name])}"
        )


def assert_unit_attr(c_attr: AttrOrPropDef, g_attr: AttrOrPropDef, op_name: str, attr_name):
    """Assert that UnitAttr fields are correctly defined."""
    if isinstance(c_attr.constr, BaseAttr) and c_attr.constr.attr == UnitAttr:
        # UnitAttrs shiould've already been parsed and converted to a standardized
        # format by the script that created the auto-generated dialects, so this
        # assertion is fine.
        assert (
            g_attr.constr.attr == UnitAttr
        ), f"The '{attr_name}' field of {op_name} is a UnitAttr, but the MLIR field is a {g_attr}."

        # UnitAttrs must also always be optional attributes or properties.
        assert isinstance(c_attr, OptionalDef), (
            "UnitAttrs must be defined as option attributes or properties. "
            f"The xDSL {op_name} operation's {attr_name} field is a UnitAttr but not optional."
        )


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to folder where the generated dialect files must be saved.",
    )

    parsed_args = parser.parse_args()
    print("Parsed arguments:")
    print(f"Save folder path: {parsed_args.save_path}")

    return parsed_args


if __name__ == "__main__":
    _parsed_args = parse_args()
    _save_path = Path(_parsed_args.save_path)

    errors = []

    for macro, dialect in all_dialects.items():
        gen_dialect_path = _save_path / f"{macro}.py"
        gen_dialect_mod = import_from_path(macro, gen_dialect_path)

        # Auto-generated dialects end with the "Dialect" suffix
        dialect_name = f"{dialect}Dialect"
        _gen_dialect = getattr(gen_dialect_mod, dialect_name)

        _catalyst_dialect = getattr(getattr(xdialects, macro), dialect)

        print(f"Verifying the '{dialect}' dialect.")

        verify_dialect(_catalyst_dialect, _gen_dialect)

        print(f"Successfully verified the '{dialect}' dialect.")
