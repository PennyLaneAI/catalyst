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


import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path

from xdsl.ir import Dialect

from catalyst.python_interface import dialects as xdialects

all_dialects = {
    "catalyst": "Catalyst",
    "mbqc": "MBQC",
    "qec": "QEC",
    "quantum": "Quantum",
}
"""Dictionary mapping the macro corresponding to a dialect its name."""


def import_from_path(module_name, file_path):
    """Dynamically import a source file/folder as a module.
    Reference: https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def verify_dialect(catalyst_dialect: Dialect, gen_dialect: Dialect):
    """Compare xDSL dialects against their TableGen definitions.

    Args:
        catalyst_dialect (xdsl.ir.Dialect): Catalyst dialect to verify
        gen_dialect (xdsl.ir.Dialect): generated dialect for comparison

    Raises:
        RuntimeError: if verification failed
    """
    assert catalyst_dialect.name == gen_dialect.name, "The name of the dialect does not match."

    c_attrs = sorted(list(catalyst_dialect.attributes), key=lambda attr: attr.__name__)
    g_attrs = sorted(list(gen_dialect.attributes), key=lambda attr: attr.__name__)
    assert len(c_attrs) == len(g_attrs), "The number of attributes does not match."

    c_ops = sorted(list(catalyst_dialect.operations), key=lambda op: op.__name__)
    g_ops = sorted(list(gen_dialect.operations), key=lambda op: op.__name__)
    assert len(c_ops) == len(g_ops), "The number of operations in the dialect does not match."

    # for
    return


def parse_args(verbose=True):
    """Parse command-line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to folder where the generated dialect files must be saved.",
    )

    parsed_args = parser.parse_args()
    if verbose:
        print("Parsed arguments:")
        print(f"Save folder path: {parsed_args.save_path}")

    return parsed_args


if __name__ == "__main__":
    parsed_args = parse_args(verbose=True)
    save_path = Path(parsed_args.save_path)

    errors = []

    for macro, dialect in all_dialects.items():
        gen_dialect_path = save_path / f"{macro}.py"
        gen_dialect_mod = import_from_path(macro, gen_dialect_path)

        # Auto-generated dialects end with the "Dialect" suffix
        dialect_name = f"{dialect}Dialect"
        gen_dialect = getattr(gen_dialect_mod, dialect_name)

        catalyst_dialect = getattr(getattr(xdialects, macro), dialect)

        print(f"Verifying the '{dialect}' dialect.")

        verify_dialect(catalyst_dialect, gen_dialect)

        print(f"Successfully verified the '{dialect}' dialect.")
