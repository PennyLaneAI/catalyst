#!/usr/bin/env python3
"""
circuit_to_qasm3.py — Convert a Qiskit circuit .py file to OpenQASM 3.0

Usage:
    python circuit_to_qasm3.py circuit.py
    python circuit_to_qasm3.py circuit.py -o output.qasm
    python circuit_to_qasm3.py circuit.py --no-opt      # skip quantum-opt
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent
_OPT_BIN  = _ROOT / "mlir" / "build" / "bin" / "quantum-opt"
_XLAT_BIN = _ROOT / "mlir" / "build" / "bin" / "quantum-translate"
_OPT_PIPELINE = "builtin.module(apply-transform-sequence, canonicalize, merge-rotations)"

PYTHONPATH_MLIR = (
    _ROOT / "mlir" / "llvm-project" / "build" / "tools" / "mlir"
    / "python_packages" / "mlir_core"
)
if str(PYTHONPATH_MLIR) not in sys.path:
    sys.path.insert(0, str(PYTHONPATH_MLIR))

# ── imports ────────────────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
except ImportError:
    sys.exit("ERROR: qiskit is not installed (pip install qiskit)")

try:
    from qiskit_importer_standalone import QiskitToCatalystImporter
except ImportError:
    sys.exit("ERROR: qiskit_importer_standalone.py not found next to this script")

_VARNAMES = ["qc", "circuit", "quantum_circuit", "qcirc", "circ"]


def load_circuit(path: Path) -> QuantumCircuit:
    ns: dict = {"__file__": str(path)}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), str(path), "exec"), ns)  # noqa: S102
    for name in _VARNAMES:
        if name in ns and isinstance(ns[name], QuantumCircuit):
            return ns[name]
    for v in ns.values():
        if isinstance(v, QuantumCircuit):
            return v
    sys.exit(f"ERROR: no QuantumCircuit found in {path}  (tried: {_VARNAMES})")


def to_mlir(qc: QuantumCircuit) -> str:
    return str(QiskitToCatalystImporter(qc).convert())


def optimize(mlir_str: str, tmp_path: str) -> str:
    if not _OPT_BIN.exists():
        sys.exit(f"ERROR: quantum-opt not found at {_OPT_BIN}")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(mlir_str)
    r = subprocess.run(
        [str(_OPT_BIN), f"--pass-pipeline={_OPT_PIPELINE}", tmp_path, "-o", tmp_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"ERROR: quantum-opt failed:\n{r.stderr.strip()}")
    with open(tmp_path, encoding="utf-8") as fh:
        return fh.read()


def translate(mlir_path: str) -> str:
    if not _XLAT_BIN.exists():
        sys.exit(f"ERROR: quantum-translate not found at {_XLAT_BIN}")
    r = subprocess.run(
        [str(_XLAT_BIN), "--mlir-to-qasm3", mlir_path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"ERROR: quantum-translate failed:\n{r.stderr.strip()}")
    return r.stdout


def main():
    parser = argparse.ArgumentParser(description="Convert a Qiskit .py circuit to QASM 3.0")
    parser.add_argument("circuit", type=Path, help="Qiskit Python circuit file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Write QASM3 to file")
    parser.add_argument("--no-opt", action="store_true", help="Skip quantum-opt optimisation pass")
    args = parser.parse_args()

    if not args.circuit.exists():
        sys.exit(f"ERROR: file not found: {args.circuit}")

    qc = load_circuit(args.circuit)
    print(f"Loaded: {qc.num_qubits} qubits, {qc.num_clbits} cbits, {len(qc.data)} gates",
          file=sys.stderr)

    mlir_str = to_mlir(qc)
    print(f"MLIR:   {len(mlir_str.splitlines())} lines", file=sys.stderr)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False, encoding="utf-8") as fh:
        tmp = fh.name

    if args.no_opt:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(mlir_str)
        print("Opt:    skipped", file=sys.stderr)
    else:
        mlir_str = optimize(mlir_str, tmp)
        print(f"Opt:    {len(mlir_str.splitlines())} lines after quantum-opt", file=sys.stderr)

    qasm3 = translate(tmp)

    if args.output:
        args.output.write_text(qasm3, encoding="utf-8")
        print(f"Output: {args.output}", file=sys.stderr)
    else:
        print(qasm3, end="")


if __name__ == "__main__":
    main()
