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

"""
OQD Compiler utilities for compiling and linking LLVM IR to ARTIQ's binary.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional


def compile_to_artiq(circuit, artiq_config, output_elf_name=None):
    """Compile a qjit-compiled circuit to ARTIQ's binary.

    This function takes a circuit compiled with target="llvmir", writes the LLVM IR
    to a file, and links it to an ARTIQ's binary.

    Args:
        circuit: A QJIT-compiled function (must be compiled with target="llvmir")
        artiq_config: Dictionary containing ARTIQ configuration:
            - kernel_ld: Path to ARTIQ kernel linker script
            - llc_path: (optional) Path to llc compiler
            - lld_path: (optional) Path to ld.lld linker
        output_elf_name: Name of the output ELF file (default: None, uses circuit function name)

    Returns:
        str: Path to the generated binary file
    """
    # Get LLVM IR text and write to file
    llvm_ir_text = circuit.qir
    circuit_name = getattr(circuit, "__name__", "circuit")
    llvm_ir_path = os.path.join(str(circuit.workspace), f"{circuit_name}.ll")
    with open(llvm_ir_path, "w", encoding="utf-8") as f:
        f.write(llvm_ir_text)
    print(f"LLVM IR file written to: {llvm_ir_path}")

    # Link to ARTIQ's binary
    if output_elf_name is None:
        output_elf_name = f"{circuit_name}.elf"

    # Output ELF file to current working directory if workspace is in /private (temp dir),
    # otherwise use workspace directory
    workspace_str = str(circuit.workspace)
    if "/private" in workspace_str:
        output_elf_path = os.path.join(os.getcwd(), output_elf_name)
    else:
        output_elf_path = os.path.join(workspace_str, output_elf_name)

    link_to_artiq_elf(
        llvm_ir_path=llvm_ir_path,
        output_elf_path=output_elf_path,
        kernel_ld=artiq_config["kernel_ld"],
        llc_path=artiq_config.get("llc_path"),
        lld_path=artiq_config.get("lld_path"),
    )

    return output_elf_path


def _validate_paths(llvm_ir_path: Path, kernel_ld: Path) -> None:
    """Validate that required input files exist."""
    if not llvm_ir_path.exists():
        raise FileNotFoundError(f"LLVM IR file not found: {llvm_ir_path}")
    if not kernel_ld.exists():
        raise FileNotFoundError(f"ARTIQ kernel.ld not found: {kernel_ld}")


def _get_tool_command(tool_path: Optional[str], default_name: str) -> str:
    """Get tool command path, validating if custom path is provided."""
    if tool_path is None:
        return default_name
    tool_path_obj = Path(tool_path)
    if not tool_path_obj.exists():
        raise FileNotFoundError(f"{default_name} not found: {tool_path}")
    return tool_path


def _run_subprocess_with_validation(
    args: list[str],
    error_prefix: str,
    file_not_found_msg: str,
    output_file: Path,
    stderr_prefix: Optional[str] = None,
) -> None:
    """Run a subprocess command and validate the output file was created.

    Args:
        args: Command arguments to pass to subprocess.run
        error_prefix: Prefix for error message on CalledProcessError
        file_not_found_msg: Message for FileNotFoundError
        output_file: Path to expected output file
        stderr_prefix: Optional prefix for stderr output
    """
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        if stderr_prefix:
            print(f"{stderr_prefix}: {result.stderr}")
    except subprocess.CalledProcessError as e:
        error_msg = f"{error_prefix} with exit code: {e.returncode}"
        if e.stderr:
            error_msg += f"\n{e.stderr}"
        raise RuntimeError(error_msg) from e
    except FileNotFoundError as exc:
        raise FileNotFoundError(file_not_found_msg) from exc

    if not output_file.exists():
        raise RuntimeError(f"{output_file.name} was not created: {output_file}")


def _compile_llvm_to_object(llvm_ir_path: Path, object_file: Path, llc_cmd: str) -> None:
    """Compile LLVM IR to object file with llc.

    Args:
        llvm_ir_path: Path to LLVM IR file
        object_file: Path to object file
        llc_cmd: Command to use for llc compiler

    Raises:
        RuntimeError: If compilation fails
        FileNotFoundError: If llc is not found
    """
    llc_args = [
        llc_cmd,
        "-mtriple=armv7-unknown-linux-gnueabihf",
        "-mcpu=cortex-a9",
        "-filetype=obj",
        "-relocation-model=pic",
        "-o",
        str(object_file),
        str(llvm_ir_path),
    ]

    print(f"[ARTIQ] Compiling with external LLC: {' '.join(llc_args)}")

    _run_subprocess_with_validation(
        args=llc_args,
        error_prefix="LLC failed",
        file_not_found_msg="llc not found. Please install LLVM or provide path via llc_path argument.",
        output_file=object_file,
        stderr_prefix="[ARTIQ] LLC stderr",
    )


def _link_object_to_elf(
    object_file: Path, output_elf_path: Path, kernel_ld: Path, lld_cmd: str
) -> None:
    """Link object file to ELF format with ld.lld.

    Args:
        object_file: Path to object file
        output_elf_path: Path to output ELF file
        kernel_ld: Path to kernel linker script
        lld_cmd: Command to use for ld.lld linker

    Raises:
        RuntimeError: If linking fails
        FileNotFoundError: If ld.lld is not found
    """
    lld_args = [
        lld_cmd,
        "-shared",
        "--eh-frame-hdr",
        "-m",
        "armelf_linux_eabi",
        "--target2=rel",
        "-T",
        str(kernel_ld),
        str(object_file),
        "-o",
        str(output_elf_path),
    ]

    print(f"[ARTIQ] Linking ELF: {' '.join(lld_args)}")

    _run_subprocess_with_validation(
        args=lld_args,
        error_prefix="LLD linking failed",
        file_not_found_msg="ld.lld not found. Please install LLVM LLD or provide path via lld_path argument.",
        output_file=output_elf_path,
        stderr_prefix="[ARTIQ] LLD stderr",
    )


# pylint: disable=too-many-arguments,too-many-positional-arguments
def link_to_artiq_elf(
    llvm_ir_path: str,
    output_elf_path: str,
    kernel_ld: str,
    llc_path: Optional[str] = None,
    lld_path: Optional[str] = None,
) -> str:
    """Link LLVM IR to ARTIQ ELF format.

    Args:
        llvm_ir_path: Path to the LLVM IR file (.ll)
        output_elf_path: Path to output ELF file
        kernel_ld: Path to ARTIQ's kernel.ld linker script
        llc_path: Path to llc (LLVM compiler). If None, uses "llc" from PATH
        lld_path: Path to ld.lld (LLVM linker). If None, uses "ld.lld" from PATH

    Returns:
        Path to the generated ELF file
    """
    llvm_ir_path_obj = Path(llvm_ir_path)
    output_elf_path_obj = Path(output_elf_path)
    kernel_ld_obj = Path(kernel_ld)

    _validate_paths(llvm_ir_path_obj, kernel_ld_obj)

    llc_cmd = _get_tool_command(llc_path, "llc")
    lld_cmd = _get_tool_command(lld_path, "ld.lld")

    object_file = output_elf_path_obj.with_suffix(".o")
    _compile_llvm_to_object(llvm_ir_path_obj, object_file, llc_cmd)
    _link_object_to_elf(object_file, output_elf_path_obj, kernel_ld_obj, lld_cmd)

    print(f"[ARTIQ] Generated ELF: {output_elf_path_obj}")

    return str(output_elf_path_obj)
