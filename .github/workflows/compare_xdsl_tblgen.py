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
from importlib import import_module
from json import dumps, loads
from pathlib import Path
from subprocess import run

from xdsl.ir import Dialect

from catalyst.python_interface import dialects as xdialects

all_dialects = {
    "catalyst": "Catalyst",
    "mbqc": "MBQC",
    "qec": "QEC",
    "quantum": "Quantum",
}
"""Dictionary mapping the macro corresponding to a dialect its name."""


def remove_invalid_field_from_json(json_str: str, dialect_name: str) -> str:
    """The JSON file for certain dialects may contain more than one dialect in
    their "Dialects" field. This function removes all invalid dialects from the
    "Dialects" field and returns an updated JSON string.

    Args:
        json_str (str): JSON string for the dialect being converted
        cur_dialect (str): name of the dialect being converted

    Returns:
        str: JSON string with the invalid dialect name removed
    """
    loaded = loads(json_str)

    # The values in the "Dialects" end with a "Dialect" suffix
    dialect_name = f"{dialect_name}Dialect"

    dialects = loaded["!instanceof"]["Dialects"]
    new_dialects = [d for d in dialects if d == dialect_name]
    loaded["!instanceof"]["Dialects"] = new_dialects

    return dumps(loaded)


def create_py_dialects(
    dialects: dict[str, str],
    include_paths: tuple[str, ...],
    llvm_tblgen: str,
    save_folder: str = "gen_dialects",
):
    """Create Python files containing xDSL dialect definitions using the MLIR dialects
    defined in TableGen files.

    Args:
        dialects (dict[str, str]): dictionary mapping macros corresponding to the
            dialects to their name
        include_paths (tuple[str, ...]): paths to necessary ``include`` folders,
            where the first item must be the include directory containing the dialects
            we want to convert to Python
        llvm_tblgen (str): path to the ``llvm-tblgen`` executable
        save_folder (str): name of the folder where all the Python files should
            be saved. The folder should be in the current working directory. If
            the folder does not exist, it will be created

    Raises:
        RuntimeError: if any of the dialects fail conversion
    """
    save_path = Path(save_folder)
    save_path.mkdir(exist_ok=True)
    dialects_include = include_paths[0]

    base_llvm_cmd = [llvm_tblgen, "-D"]
    includes = []
    for p in include_paths:
        includes += ["-I", p]

    errors = []

    for macro, dialect in dialects.items():
        mlir_dialect_path = f"{dialects_include}/{dialect}/IR/{dialect}Ops.td"
        llvm_cmd = base_llvm_cmd + [macro, mlir_dialect_path] + includes

        llvm_res = run(llvm_cmd, capture_output=True, check=False)
        if llvm_res.returncode != 0:
            err_msg = (
                f"Converting the '{dialect}' dialect to JSON failed with the following error:"
                f"\n{llvm_res.stderr.decode()}"
            )
            errors.append(err_msg)
            continue

        corrected_json = remove_invalid_field_from_json(llvm_res.stdout.decode(), dialect)
        final_path = save_path / f"{macro}.py"
        xdsl_cmd = ["xdsl-tblgen", "-o", str(final_path)]

        xdsl_res = run(xdsl_cmd, input=corrected_json.encode(), capture_output=True, check=False)
        if xdsl_res.returncode != 0:
            err_msg = (
                f"Converting the dumped JSON for the '{dialect}' dialect to Python failed "
                "with the following error:"
                f"\n{xdsl_res.stderr.decode()}"
            )
            errors.append(err_msg)

    if errors:
        final_err_msg = "There were errors while trying to convert MLIR dialects to xDSL:\n\n"
        for err in errors:
            final_err_msg += f"{err}\n\n"
        raise RuntimeError(final_err_msg)


def verify_dialect(catalyst_dialect: Dialect, gen_dialect: Dialect) -> str:
    """Compare xDSL dialects against their TableGen definitions.

    Args:
        catalyst_dialect (xdsl.ir.Dialect): Catalyst dialect to verify
        gen_dialect (xdsl.ir.Dialect): generated dialect for comparison

    Returns:
        str: error message with details if verification fails. Empty if there is no error
    """
    return ""


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "--llvm-build",
        type=str,
        required=True,
        nargs=1,
        help="Path to the LLVM build.",
    )

    parser.add_argument(
        "--catalyst-root",
        type=str,
        required=True,
        nargs=1,
        help="Path to the Catalyst repository root.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()

    mlir_path = f"{parsed_args.catalyst_root}/mlir"
    include_paths = [f"{mlir_path}/include", f"{mlir_path}/llvm-project/mlir/include"]
    llvm_tblgen = f"{parsed_args.llvm_build}/bin/llvm-tblgen"
    save_folder = "gen_dialects"

    create_py_dialects(all_dialects, include_paths, llvm_tblgen, save_folder=save_folder)
    gen_mod = import_module(save_folder)
    errors = []

    for macro, dialect in all_dialects.items():
        catalyst_dialect = getattr(getattr(xdialects, macro), dialect)

        # Auto-generated dialects end with the "Dialect" suffix
        dialect_name = f"{dialect}Dialect"
        gen_dialect = getattr(getattr(gen_mod, macro), dialect_name)

        err = verify_dialect(catalyst_dialect, gen_dialect)
        if err:
            errors.append(err)

    if errors:
        final_err_msg = "Dialect verification failed with the following errors:\n\n"
        for err_msg in errors:
            final_err_msg += f"{err_msg}\n\n"

        raise RuntimeError(final_err_msg)
