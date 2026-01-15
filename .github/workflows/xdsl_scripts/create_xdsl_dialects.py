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
from json import dumps, loads
from pathlib import Path
from re import sub
from subprocess import list2cmdline, run

all_dialects = {
    "catalyst": "Catalyst",
    "mbqc": "MBQC",
    "qec": "QEC",
    "quantum": "Quantum",
}
"""Dictionary mapping the macro corresponding to a dialect its name."""


def create_py_dialect(
    dialect_macro: str,
    dialect_name: str,
    include_paths: tuple[str, ...],
    llvm_tblgen: str,
    save_path: Path,
):
    """Create a Python file containing the xDSL dialect using an MLIR dialect.

    Args:
        dialect_macro (str): the macro for the dialect
        dialect_name (str): the name of the dialect
        include_paths (tuple[str, ...]): paths to necessary ``include`` folders,
            where the first item must be the include directory containing the dialects
            we want to convert to Python
        llvm_tblgen (str): path to the ``llvm-tblgen`` executable
        save_path (Path): path to folder where the dialect file should be saved

    Raises:
        RuntimeError: if conversion fails
    """
    print(f"Converting the '{dialect_name}' dialect.")

    dialects_include = include_paths[0]
    mlir_dialect_path = f"{dialects_include}/{dialect_name}/IR/{dialect_name}Ops.td"

    base_llvm_cmd = [llvm_tblgen, "--dump-json"]
    for p in include_paths:
        base_llvm_cmd += ["-I", p]
    llvm_cmd = base_llvm_cmd + ["-D", dialect_macro, mlir_dialect_path]

    print(f"Generating JSON using the following command: '{list2cmdline(llvm_cmd)}'")

    llvm_res = run(llvm_cmd, capture_output=True, check=False)
    if llvm_res.returncode != 0:
        raise RuntimeError(
            f"Generating JSON for the '{dialect_name}' dialect failed with the following error:"
            f"\n{llvm_res.stderr.decode()}"
        )

    corrected_json = remove_invalid_fields_from_json(llvm_res.stdout.decode(), dialect_name)
    final_path = save_path / f"{dialect_macro}.py"
    xdsl_cmd = ["xdsl-tblgen", "-o", str(final_path)]

    print(f"Generating Python dialect using the following command: '{list2cmdline(xdsl_cmd)}'")

    xdsl_res = run(xdsl_cmd, input=corrected_json.encode(), capture_output=True, check=False)
    if xdsl_res.returncode != 0:
        raise RuntimeError(
            f"Converting the dumped JSON for the '{dialect_name}' dialect to Python failed "
            "with the following error:"
            f"\n{xdsl_res.stderr.decode()}"
        )

    print(f"'{dialect_name}' dialect converted successfully and saved to '{str(final_path)}'.")
    print(f"Stripping unnecessary operation definition details from the '{dialect_name}' dialect.")

    strip_op_defs(final_path)

    print("Stripped operation definition details successfully.\n")


def remove_invalid_fields_from_json(json_str: str, dialect_name: str) -> str:
    """The JSON file for certain dialects may contain more than one dialect in
    their "Dialect" field. This function removes all invalid dialects from the
    "Dialect" field and returns an updated JSON string. Additionally, we don't
    want the auto-generated dialects to include assembly formats, which will
    also be removed.

    Args:
        json_str (str): JSON string for the dialect being converted
        cur_dialect (str): name of the dialect being converted

    Returns:
        str: JSON string with the invalid dialect name removed
    """
    # Normalize the dialect name
    json_str = sub(f"{dialect_name}_Dialect", f"{dialect_name}Dialect", json_str)

    loaded = loads(json_str)

    dialects = loaded["!instanceof"]["Dialect"]
    new_dialects = [d for d in dialects if dialect_name in d]
    loaded["!instanceof"]["Dialect"] = new_dialects

    for val in loaded.values():
        if isinstance(val, dict):
            _ = val.pop("assemblyFormat", None)
            _ = val.pop("hasCustomAssemblyFormat", None)

    return dumps(loaded)


def strip_op_defs(file_path: Path):
    """Strip unnecessary details from the operation definitions of
    all classes within the provided file. ``xdsl-tblgen`` does not
    always produce executable code since it generates constraints
    that are sometimes invalid. This function replaces all constraints
    with ``AnyAttr()``, since the constraints will not be used during
    verification.

    Args:
        file_path (pathlib.Path): path to the Python file being stripped
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove operand/property/attribute/result constraints
    search_pattern = r"((operand|prop|result|attr)_def\().*(\)\n)"
    replace_pattern = r"\1AnyAttr()\3"
    content = sub(search_pattern, replace_pattern, content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_args():
    """Parse command-line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "--llvm-build",
        type=str,
        required=True,
        help="Path to the LLVM build.",
    )

    parser.add_argument(
        "--catalyst-root",
        type=str,
        required=True,
        help="Path to the Catalyst repository root.",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to folder where the generated dialect files must be saved.",
    )

    parsed_args = parser.parse_args()

    print("Parsed arguments:")
    print(f"LLVM build directory: {parsed_args.llvm_build}")
    print(f"Catalyst root: {parsed_args.catalyst_root}")
    print(f"Save folder path: {parsed_args.save_path}\n")

    return parsed_args


if __name__ == "__main__":
    _parsed_args = parse_args()

    _mlir_path = f"{_parsed_args.catalyst_root}/mlir"
    _include_paths = [f"{_mlir_path}/include", f"{_mlir_path}/llvm-project/mlir/include"]
    _llvm_tblgen = f"{_parsed_args.llvm_build}/bin/llvm-tblgen"
    _save_path = Path(_parsed_args.save_path)

    if not _save_path.exists():
        _save_path.mkdir()

    for macro, dialect in all_dialects.items():
        create_py_dialect(macro, dialect, _include_paths, _llvm_tblgen, _save_path)
