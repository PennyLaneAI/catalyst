#!/usr/bin/env python3
"""
QCC Simulation Validator
Quantum Computing Compiler - OpenQASM 3 Translation Pass

CI/CD-style validator for the full QCC compilation pipeline:
  [1] Frontend        Load .qasm / Qiskit .py / PennyLane .py → QuantumCircuit
  [2] MLIR Gen        QiskitToCatalystImporter.convert()       → MLIR string
  [3] Optimize        quantum-opt passes                       → optimized MLIR
  [4] Translate       quantum-translate --mlir-to-qasm3        → QASM3 string
  [5] Struct Valid    Header / qubit count / gate checks
  [6] Sim Valid       Aer simulation + Hellinger distance       (optional)

Usage:
  python qcc_simulator.py circuit.qasm
  python qcc_simulator.py circuit.py --lang pennylane
  python qcc_simulator.py circuit.py --lang qiskit
  python qcc_simulator.py --batch circuits/
  python qcc_simulator.py --batch circuits/ --simulation
  python qcc_simulator.py --batch circuits/ --verbose
  python qcc_simulator.py --batch circuits/ --report out.txt
  python qcc_simulator.py circuit.qasm --no-color
"""

# ── Section 1: Imports + path bootstrap ───────────────────────────────────────
import argparse
import inspect
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

_ROOT = Path(__file__).parent.resolve()
_MLIR_CORE = (
    _ROOT / "mlir" / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core"
)
if _MLIR_CORE.exists() and str(_MLIR_CORE) not in sys.path:
    sys.path.insert(0, str(_MLIR_CORE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from qiskit import QuantumCircuit, transpile

    _QISKIT_OK = True
except ImportError:
    _QISKIT_OK = False

try:
    from qiskit_importer_standalone import QiskitToCatalystImporter

    _IMPORTER_OK = True
except ImportError:
    _IMPORTER_OK = False

try:
    import qiskit.qasm3 as _qasm3_mod
    from qiskit_aer import AerSimulator

    _AER_OK = True
except ImportError:
    _AER_OK = False

try:
    import pennylane as qml

    _PL_OK = True
except ImportError:
    _PL_OK = False


# ── Section 2: Constants ───────────────────────────────────────────────────────
_WIDTH = 72

_OPT_BIN = _ROOT / "mlir" / "build" / "bin" / "quantum-opt"
_XLAT_BIN = _ROOT / "mlir" / "build" / "bin" / "quantum-translate"

_OPT_PIPELINE = "builtin.module(apply-transform-sequence, canonicalize, merge-rotations)"

# (index 0-based, short_name, display_name)
STAGES = [
    (0, "frontend", "Frontend: Circuit Import"),
    (1, "mlir_gen", "MLIR Generation"),
    (2, "opt", "Optimization (quantum-opt)"),
    (3, "translate", "QASM3 Translation"),
    (4, "validate", "Structural Validation"),
    (5, "simulate", "Simulation Validation (Aer)"),
]

_QISKIT_VARNAMES = ["qc", "circuit", "quantum_circuit", "qcirc", "circ"]
_PL_VARNAMES = ["circuit", "qnode", "ansatz", "kernel", "qfunc"]

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


# ── Section 3: Renderer ────────────────────────────────────────────────────────
class Renderer:
    """All ASCII/CI-style output. Writes to stdout + optional report file."""

    def __init__(self, report_path: Optional[Path] = None, color: bool = True):
        self._report_fh = open(report_path, "w", encoding="utf-8") if report_path else None
        self._color_on = color and sys.stdout.isatty()

    # ── low-level ──────────────────────────────────────────────────────────────

    def _emit(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()
        if self._report_fh:
            self._report_fh.write(_strip_ansi(text))
            self._report_fh.flush()

    def _c(self, text: str, code: str) -> str:
        return f"{code}{text}{_RESET}" if self._color_on else text

    # ── structural elements ────────────────────────────────────────────────────

    def banner(self) -> None:
        inner = _WIDTH - 2
        self._emit(f"┌{'─' * inner}┐\n")
        self._emit(
            f"│  {self._c('QCC SIMULATION VALIDATOR', _BOLD):<{inner - 2 + (len(_BOLD) + len(_RESET) if self._color_on else 0)}}│\n"
        )
        self._emit(
            f"│  {'Quantum Computing Compiler - OpenQASM 3 Translation Pass':<{inner - 2}}│\n"
        )
        self._emit(f"└{'─' * inner}┘\n\n")

    def circuit_header(self, name: str) -> None:
        self._emit(f"Circuit: {self._c(name, _BOLD)}\n")
        self._emit("─" * _WIDTH + "\n")

    def blank(self) -> None:
        self._emit("\n")

    # ── stage lines ────────────────────────────────────────────────────────────

    def _stage_prefix(self, idx: int, total: int, name: str) -> str:
        return f"[STAGE {idx}/{total}]  {name}"

    def stage_running(self, idx: int, total: int, name: str) -> None:
        prefix = self._stage_prefix(idx, total, name)
        raw_status = "RUNNING..."
        pad = max(1, _WIDTH - len(prefix) - len(raw_status))
        self._emit(prefix + " " * pad + self._c(raw_status, _CYAN) + "\n")

    def stage_pass(self, idx: int, total: int, name: str, elapsed: float) -> None:
        prefix = self._stage_prefix(idx, total, name)
        raw_status = f"PASS  {elapsed:.2f}s"
        pad = max(1, _WIDTH - len(prefix) - len(raw_status))
        self._emit(prefix + " " * pad + self._c(raw_status, _GREEN) + "\n\n")

    def stage_fail(self, idx: int, total: int, name: str, elapsed: float) -> None:
        prefix = self._stage_prefix(idx, total, name)
        raw_status = f"FAIL  {elapsed:.2f}s"
        pad = max(1, _WIDTH - len(prefix) - len(raw_status))
        self._emit(prefix + " " * pad + self._c(raw_status, _RED) + "\n\n")

    def stage_skip(self, idx: int, total: int, name: str) -> None:
        prefix = self._stage_prefix(idx, total, name)
        raw_status = "SKIP"
        pad = max(1, _WIDTH - len(prefix) - len(raw_status))
        self._emit(prefix + " " * pad + self._c(raw_status, _YELLOW) + "\n\n")

    # ── info / error bullets ───────────────────────────────────────────────────

    def info(self, label: str, value: str) -> None:
        self._emit(f"  {self._c('▶', _CYAN)}  {label:<10}: {value}\n")

    def error_detail(self, message: str) -> None:
        for line in message.strip().splitlines():
            self._emit(f"  {self._c('!', _RED)}  {line}\n")

    # ── verbose block ──────────────────────────────────────────────────────────

    def verbose_block(self, label: str, content: str, max_lines: int = 40) -> None:
        self._emit(f"  {self._c(f'--- {label} ---', _YELLOW)}\n")
        lines = content.splitlines()
        for line in lines[:max_lines]:
            self._emit(f"  | {line}\n")
        if len(lines) > max_lines:
            self._emit(f"  | ... ({len(lines) - max_lines} more lines)\n")
        self._emit(f"  ---\n\n")

    # ── circuit result footer ──────────────────────────────────────────────────

    def circuit_result(self, passed: bool, total_elapsed: float) -> None:
        self._emit("═" * _WIDTH + "\n")
        if passed:
            mark = self._c("✓ PASS", _GREEN + _BOLD)
        else:
            mark = self._c("✗ FAIL", _RED + _BOLD)
        raw_mark = "✓ PASS" if passed else "✗ FAIL"
        pad = max(
            1,
            _WIDTH
            - len("CIRCUIT RESULT:  ")
            - len(raw_mark)
            - len(f"    Total: {total_elapsed:.2f}s"),
        )
        self._emit(f"CIRCUIT RESULT:  {mark}{' ' * pad}Total: {total_elapsed:.2f}s\n")
        self._emit("═" * _WIDTH + "\n\n")

    # ── batch summary ──────────────────────────────────────────────────────────

    def batch_summary(self, results: List[Dict]) -> None:
        self._emit("═" * _WIDTH + "\n")
        self._emit(self._c("BATCH SUMMARY\n", _BOLD))
        self._emit("─" * _WIDTH + "\n")

        col_name = 30
        n_stages = len(results[0]["stages"]) if results else 6
        stage_cols = " ".join(f"{i+1:>2}" for i in range(n_stages))
        header = f"  {'Circuit':<{col_name}} {stage_cols}    Result     Time\n"
        self._emit(header)
        self._emit("  " + "─" * (_WIDTH - 2) + "\n")

        pass_count = fail_count = 0
        for r in results:
            marks = []
            for s in r["stages"]:
                if s is True:
                    marks.append(self._c("✓", _GREEN))
                elif s is False:
                    marks.append(self._c("✗", _RED))
                else:
                    marks.append(self._c("-", _YELLOW))

            result_str = self._c("PASS", _GREEN) if r["passed"] else self._c("FAIL", _RED)
            raw_result = "PASS" if r["passed"] else "FAIL"
            if r["passed"]:
                pass_count += 1
            else:
                fail_count += 1

            name = r["name"][:col_name]
            elapsed_str = f"{r['elapsed']:.2f}s"
            self._emit(
                f"  {name:<{col_name}} "
                + "  ".join(marks)
                + f"    {result_str}      {elapsed_str}\n"
            )

        total = len(results)
        self._emit("  " + "─" * (_WIDTH - 2) + "\n")
        passed_s = self._c(str(pass_count), _GREEN)
        failed_s = self._c(str(fail_count), _RED)
        self._emit(f"  Total: {total} circuits | {passed_s} PASS | {failed_s} FAIL\n")
        self._emit("═" * _WIDTH + "\n")

    def close(self) -> None:
        if self._report_fh:
            self._report_fh.close()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


# ── Section 4: StageResult + inlined utilities ────────────────────────────────


class StageResult(NamedTuple):
    passed: bool
    data: Any  # stage-specific payload
    message: str  # human-readable detail or error


def hellinger_distance(dict1: dict, dict2: dict, shots: int) -> float:
    """Compute Hellinger distance between two shot-count distributions."""
    p1 = {k: v / shots for k, v in dict1.items()}
    p2 = {k: v / shots for k, v in dict2.items()}
    keys = set(p1.keys()).union(set(p2.keys()))
    distance_sq = 0.0
    for k in keys:
        distance_sq += (math.sqrt(p1.get(k, 0.0)) - math.sqrt(p2.get(k, 0.0))) ** 2
    return math.sqrt(distance_sq) / math.sqrt(2)


def aggregate_by_hamming_weight(counts_dict: dict) -> dict:
    """Aggregate measurement counts by Hamming weight (number of '1' bits)."""
    agg: Dict[str, int] = {}
    for k, v in counts_dict.items():
        weight = str(k.count("1"))
        agg[weight] = agg.get(weight, 0) + v
    return agg


# ── Section 5: Stage 1 — Frontend ─────────────────────────────────────────────


def _extract_qiskit_circuit(ns: dict):
    """Scan an exec'd namespace for a QuantumCircuit object."""
    for name in _QISKIT_VARNAMES:
        if name in ns and isinstance(ns[name], QuantumCircuit):
            return ns[name]
    for v in ns.values():
        if isinstance(v, QuantumCircuit):
            return v
    return None


def _pennylane_to_qiskit(path: Path):
    """Load a PennyLane .py file and convert the QNode to a QuantumCircuit."""
    ns: Dict = {"__file__": str(path)}
    with open(path, encoding="utf-8") as fh:
        exec(compile(fh.read(), str(path), "exec"), ns)  # noqa: S102

    qnode = None
    for name in _PL_VARNAMES:
        obj = ns.get(name)
        if obj is not None and isinstance(obj, qml.QNode):
            qnode = obj
            break
    if qnode is None:
        for v in ns.values():
            if isinstance(v, qml.QNode):
                qnode = v
                break
    if qnode is None:
        return None, None

    # Build call arguments: zero-fill required params
    sig = inspect.signature(qnode.func)
    args = [0.0 for p in sig.parameters.values() if p.default is inspect.Parameter.empty]
    tape = qml.workflow.construct_tape(qnode)(*args)
    qasm2_str = tape.to_openqasm(measure_all=True, rotations=False)
    qc = QuantumCircuit.from_qasm_str(qasm2_str)
    return qc, args


def run_stage1_frontend(path: Path, lang: str) -> StageResult:
    if not _QISKIT_OK:
        return StageResult(False, None, "qiskit is not installed")

    if lang == "qasm":
        try:
            qc = QuantumCircuit.from_qasm_file(str(path))
            detail = f"{qc.num_qubits} qubits, {qc.num_clbits} cbits, {len(qc.data)} gates"
            return StageResult(True, qc, detail)
        except Exception as exc:
            return StageResult(False, None, str(exc))

    elif lang == "qiskit":
        try:
            ns: Dict = {"__file__": str(path)}
            with open(path, encoding="utf-8") as fh:
                exec(compile(fh.read(), str(path), "exec"), ns)  # noqa: S102
            qc = _extract_qiskit_circuit(ns)
            if qc is None:
                return StageResult(
                    False, None, f"No QuantumCircuit found. Searched names: {_QISKIT_VARNAMES}"
                )
            detail = f"{qc.num_qubits} qubits, {qc.num_clbits} cbits, {len(qc.data)} gates"
            return StageResult(True, qc, detail)
        except Exception as exc:
            return StageResult(False, None, str(exc))

    elif lang == "pennylane":
        if not _PL_OK:
            return StageResult(False, None, "pennylane is not installed")
        try:
            qc, args = _pennylane_to_qiskit(path)
            if qc is None:
                return StageResult(False, None, f"No QNode found. Searched names: {_PL_VARNAMES}")
            note = ""
            if args:
                note = f" (params set to {[0.0]*len(args)})"
            detail = (
                f"{qc.num_qubits} qubits, {qc.num_clbits} cbits, " f"{len(qc.data)} gates{note}"
            )
            return StageResult(True, qc, detail)
        except Exception as exc:
            return StageResult(False, None, str(exc))

    return StageResult(False, None, f"Unknown language: {lang!r}")


# ── Section 6: Stage 2 — MLIR Generation ──────────────────────────────────────


def run_stage2_mlir(qc) -> StageResult:
    if not _IMPORTER_OK:
        return StageResult(False, None, "qiskit_importer_standalone not available")
    try:
        module = QiskitToCatalystImporter(qc).convert()
        mlir_str = str(module)
        lines = mlir_str.splitlines()
        ops = sum(1 for ln in lines if "=" in ln)
        detail = f"{len(lines)} lines, {ops} operations"
        return StageResult(True, mlir_str, detail)
    except Exception as exc:
        return StageResult(False, None, str(exc))


# ── Section 7: Stage 3 — Optimization ─────────────────────────────────────────


def run_stage3_optimize(mlir_str: str) -> StageResult:
    if not _OPT_BIN.exists():
        return StageResult(False, (None, None), f"quantum-opt not found: {_OPT_BIN}")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(mlir_str)
            tmp_path = fh.name

        cmd = [
            str(_OPT_BIN),
            f"--pass-pipeline={_OPT_PIPELINE}",
            tmp_path,
            "-o",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr.strip()
            return StageResult(
                False, (tmp_path, None), f"quantum-opt exited {result.returncode}: {err[:400]}"
            )

        with open(tmp_path, encoding="utf-8") as fh:
            opt_mlir = fh.read()
        lines = opt_mlir.splitlines()
        ops = sum(1 for ln in lines if "=" in ln)
        detail = f"{len(lines)} lines, {ops} ops after optimization"
        return StageResult(True, (tmp_path, opt_mlir), detail)

    except Exception as exc:
        return StageResult(False, (tmp_path, None), str(exc))


# ── Section 8: Stage 4 — QASM3 Translation ────────────────────────────────────


def run_stage4_translate(tmp_mlir_path: str) -> StageResult:
    if not _XLAT_BIN.exists():
        return StageResult(False, None, f"quantum-translate not found: {_XLAT_BIN}")
    try:
        cmd = [str(_XLAT_BIN), "--mlir-to-qasm3", tmp_mlir_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr.strip()
            return StageResult(
                False, None, f"quantum-translate exited {result.returncode}: {err[:400]}"
            )
        qasm3 = result.stdout
        lines = qasm3.splitlines()
        boilerplate = {"OPENQASM", "include", "gate", "qubit", "bit", "//", ""}
        gate_lines = [
            ln
            for ln in lines
            if ln.strip() and not any(ln.strip().startswith(b) for b in boilerplate)
        ]
        detail = f"{len(lines)} lines, {len(gate_lines)} instructions"
        return StageResult(True, qasm3, detail)
    except Exception as exc:
        return StageResult(False, None, str(exc))


# ── Section 9: Stage 5 — Structural Validation ────────────────────────────────


def run_stage5_validate(qasm3: str, original_qc) -> StageResult:
    errors = []

    if "OPENQASM 3.0" not in qasm3:
        errors.append("Missing 'OPENQASM 3.0' header")

    if "stdgates.inc" not in qasm3:
        errors.append("Missing 'include \"stdgates.inc\"'")

    qubit_counts = re.findall(r"qubit\[(\d+)\]", qasm3)
    total_declared = sum(int(x) for x in qubit_counts)
    if total_declared != original_qc.num_qubits:
        errors.append(
            f"Qubit count mismatch: expected {original_qc.num_qubits}, "
            f"declared {total_declared}"
        )

    # At least one non-boilerplate line
    non_header = [
        ln
        for ln in qasm3.splitlines()
        if ln.strip()
        and not ln.startswith("//")
        and not ln.startswith("OPENQASM")
        and not ln.startswith("include")
        and not ln.startswith("gate")
    ]
    if not non_header:
        errors.append("QASM3 output has no qubit declarations or gate instructions")

    if errors:
        return StageResult(False, None, "; ".join(errors))

    checks = [
        "header OK",
        "stdgates OK",
        f"qubits={total_declared}",
        f"{len(non_header)} declarations/instructions",
    ]
    return StageResult(True, None, ", ".join(checks))


# ── Section 10: Stage 6 — Simulation Validation ───────────────────────────────


def run_stage6_simulate(qasm3: str, original_qc) -> StageResult:
    if not _AER_OK:
        return StageResult(
            False,
            None,
            "qiskit-aer or qiskit.qasm3 not available (install with: pip install qiskit-aer)",
        )

    if re.search(r"^\s*if\s*\(", qasm3, re.MULTILINE) or re.search(
        r"^\s*for\s+", qasm3, re.MULTILINE
    ):
        return StageResult(
            True,
            "COND_SKIP",
            "Skipped — circuit contains classical control flow "
            "(QASM3 simulation not supported for if/for blocks)",
        )

    if original_qc.num_clbits == 0:
        return StageResult(True, None, "Skipped — no classical bits (circuit has no measurements)")

    SHOTS = 10_000
    try:
        sim = AerSimulator()
        qc_t = transpile(original_qc, sim)
        res1 = sim.run(qc_t, shots=SHOTS).result()
        counts1 = res1.get_counts()

        qc2 = _qasm3_mod.loads(qasm3)
        qc2_t = transpile(qc2, sim)
        res2 = sim.run(qc2_t, shots=SHOTS).result()
        counts2 = res2.get_counts()

        agg1 = aggregate_by_hamming_weight(counts1)
        agg2 = aggregate_by_hamming_weight(counts2)
        dist = hellinger_distance(agg1, agg2, SHOTS)

        if dist > 0.1:
            return StageResult(False, dist, f"Hellinger distance {dist:.4f} > 0.1 threshold")
        return StageResult(True, dist, f"Hellinger distance {dist:.4f} ≤ 0.1 — distributions match")

    except Exception as exc:
        return StageResult(
            False, None, f"Simulation error (translation may still be valid): {str(exc)[:300]}"
        )


# ── Section 11: Helpers ────────────────────────────────────────────────────────


def _cleanup_tmp(path: Optional[str]) -> None:
    if path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _detect_lang(path: Path) -> str:
    if path.suffix == ".qasm":
        return "qasm"
    if path.suffix == ".py":
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if "pennylane" in content or "@qml.qnode" in content:
                return "pennylane"
        except OSError:
            pass
        return "qiskit"
    return "qasm"


# ── Section 12: PipelineRunner ────────────────────────────────────────────────


class PipelineRunner:
    """Orchestrates the 6-stage pipeline for a single circuit file."""

    def __init__(self, renderer: Renderer, verbose: bool = False, no_simulation: bool = False):
        self.R = renderer
        self.verbose = verbose
        self.no_simulation = no_simulation

    def run(self, path: Path, lang: str) -> Dict:
        R = self.R
        R.circuit_header(path.name)

        total_stages = 5 if self.no_simulation else 6
        stages: List[Optional[bool]] = [None] * total_stages
        t_total = time.perf_counter()
        tmp_path = None

        # ── Stage 1 ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        R.stage_running(1, total_stages, STAGES[0][2])
        sr = run_stage1_frontend(path, lang)
        elapsed = time.perf_counter() - t0

        if sr.passed:
            qc = sr.data
            R.info("Source", str(path))
            R.info("Circuit", sr.message)
            R.stage_pass(1, total_stages, STAGES[0][2], elapsed)
            stages[0] = True
        else:
            R.error_detail(sr.message)
            R.stage_fail(1, total_stages, STAGES[0][2], elapsed)
            stages[0] = False
            for i in range(1, total_stages):
                R.stage_skip(i + 1, total_stages, STAGES[i][2])
            R.circuit_result(False, time.perf_counter() - t_total)
            return _make_result(path.name, stages, False, time.perf_counter() - t_total)

        # ── Stage 2 ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        R.stage_running(2, total_stages, STAGES[1][2])
        sr = run_stage2_mlir(qc)
        elapsed = time.perf_counter() - t0

        if sr.passed:
            mlir_str = sr.data
            R.info("MLIR", sr.message)
            R.stage_pass(2, total_stages, STAGES[1][2], elapsed)
            stages[1] = True
            if self.verbose:
                R.verbose_block("MLIR OUTPUT", mlir_str)
        else:
            R.error_detail(sr.message)
            R.stage_fail(2, total_stages, STAGES[1][2], elapsed)
            stages[1] = False
            for i in range(2, total_stages):
                R.stage_skip(i + 1, total_stages, STAGES[i][2])
            R.circuit_result(False, time.perf_counter() - t_total)
            return _make_result(path.name, stages, False, time.perf_counter() - t_total)

        # ── Stage 3 ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        R.stage_running(3, total_stages, STAGES[2][2])
        sr = run_stage3_optimize(mlir_str)
        elapsed = time.perf_counter() - t0

        if sr.passed:
            tmp_path, opt_mlir = sr.data
            R.info("Opt MLIR", sr.message)
            R.stage_pass(3, total_stages, STAGES[2][2], elapsed)
            stages[2] = True
        else:
            tmp_path = sr.data[0] if sr.data else None
            R.error_detail(sr.message)
            R.stage_fail(3, total_stages, STAGES[2][2], elapsed)
            stages[2] = False
            for i in range(3, total_stages):
                R.stage_skip(i + 1, total_stages, STAGES[i][2])
            _cleanup_tmp(tmp_path)
            R.circuit_result(False, time.perf_counter() - t_total)
            return _make_result(path.name, stages, False, time.perf_counter() - t_total)

        # ── Stage 4 ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        R.stage_running(4, total_stages, STAGES[3][2])
        sr = run_stage4_translate(tmp_path)
        elapsed = time.perf_counter() - t0
        _cleanup_tmp(tmp_path)
        tmp_path = None

        if sr.passed:
            qasm3 = sr.data
            R.info("QASM3", sr.message)
            R.stage_pass(4, total_stages, STAGES[3][2], elapsed)
            stages[3] = True
            if self.verbose:
                R.verbose_block("QASM3 OUTPUT", qasm3)
        else:
            R.error_detail(sr.message)
            R.stage_fail(4, total_stages, STAGES[3][2], elapsed)
            stages[3] = False
            for i in range(4, total_stages):
                R.stage_skip(i + 1, total_stages, STAGES[i][2])
            R.circuit_result(False, time.perf_counter() - t_total)
            return _make_result(path.name, stages, False, time.perf_counter() - t_total)

        # ── Stage 5 ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        R.stage_running(5, total_stages, STAGES[4][2])
        sr = run_stage5_validate(qasm3, qc)
        elapsed = time.perf_counter() - t0

        if sr.passed:
            R.info("Checks", sr.message)
            R.stage_pass(5, total_stages, STAGES[4][2], elapsed)
            stages[4] = True
        else:
            R.error_detail(sr.message)
            R.stage_fail(5, total_stages, STAGES[4][2], elapsed)
            stages[4] = False
            if not self.no_simulation:
                R.stage_skip(6, total_stages, STAGES[5][2])
            R.circuit_result(False, time.perf_counter() - t_total)
            return _make_result(path.name, stages, False, time.perf_counter() - t_total)

        # ── Stage 6 (optional) ────────────────────────────────────────────────
        if self.no_simulation:
            total_elapsed = time.perf_counter() - t_total
            overall = all(s is True for s in stages)
            R.circuit_result(overall, total_elapsed)
            return _make_result(path.name, stages, overall, total_elapsed)

        t0 = time.perf_counter()
        R.stage_running(6, total_stages, STAGES[5][2])
        sr = run_stage6_simulate(qasm3, qc)
        elapsed = time.perf_counter() - t0

        if sr.data == "COND_SKIP":
            R.info("Result", sr.message)
            R.stage_skip(6, total_stages, STAGES[5][2])
            stages[5] = None
        elif sr.passed:
            R.info("Result", sr.message)
            R.stage_pass(6, total_stages, STAGES[5][2], elapsed)
            stages[5] = True
        else:
            R.error_detail(sr.message)
            R.stage_fail(6, total_stages, STAGES[5][2], elapsed)
            stages[5] = False

        total_elapsed = time.perf_counter() - t_total
        overall = all(s is True for s in stages if s is not None)
        R.circuit_result(overall, total_elapsed)
        return _make_result(path.name, stages, overall, total_elapsed)


def _make_result(name: str, stages: list, passed: bool, elapsed: float) -> Dict:
    return {"name": name, "stages": stages, "passed": passed, "elapsed": elapsed}


# ── Section 13: BatchRunner ───────────────────────────────────────────────────

_SKIP_PREFIXES = ("_", "conftest", "test_", "benchmark_", "random_circuit")
_SUPPORTED_EXTS = {".qasm", ".py"}


class BatchRunner:
    """Discover and process all circuits in a directory."""

    def __init__(self, renderer: Renderer, verbose: bool = False, no_simulation: bool = False):
        self.pipeline = PipelineRunner(renderer, verbose, no_simulation)
        self.renderer = renderer

    def run(self, directory: Path) -> int:
        files = sorted(
            f
            for f in directory.iterdir()
            if f.suffix in _SUPPORTED_EXTS and not any(f.name.startswith(p) for p in _SKIP_PREFIXES)
        )
        if not files:
            self.renderer._emit(f"No circuit files found in {directory}\n")
            return 1

        self.renderer._emit(f"Found {len(files)} circuit(s) in {directory}\n\n")
        all_results = []
        for f in files:
            lang = _detect_lang(f)
            result = self.pipeline.run(f, lang)
            all_results.append(result)

        self.renderer.batch_summary(all_results)
        failed = sum(1 for r in all_results if not r["passed"])
        return 0 if failed == 0 else 1


# ── Section 14: CLI entry point ───────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="qcc_simulator",
        description="QCC Simulation Validator — CI/CD-style quantum compilation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python qcc_simulator.py circuit.qasm\n"
            "  python qcc_simulator.py circuit.py --lang pennylane\n"
            "  python qcc_simulator.py --batch test_translation_qasm3/qasm3_circuits/\n"
            "  python qcc_simulator.py --batch circuits/ --simulation\n"
            "  python qcc_simulator.py --batch circuits/ --verbose\n"
            "  python qcc_simulator.py --batch circuits/ --report report.txt\n"
        ),
    )
    parser.add_argument(
        "circuit", nargs="?", type=Path, help="Path to a single circuit file (.qasm or .py)"
    )
    parser.add_argument(
        "--lang",
        choices=["qasm", "qiskit", "pennylane"],
        default=None,
        help="Force language (auto-detected from extension/content if omitted)",
    )
    parser.add_argument("--batch", type=Path, metavar="DIR", help="Run all circuits in DIR")
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Enable Stage 6 Aer simulation (disabled by default)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print intermediate MLIR and QASM3 content"
    )
    parser.add_argument(
        "--report", type=Path, metavar="FILE", help="Write output to FILE (ANSI codes stripped)"
    )
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.circuit is None and args.batch is None:
        print("Error: specify a circuit file or --batch <dir>", file=sys.stderr)
        print("Run with --help for usage.", file=sys.stderr)
        return 2

    renderer = Renderer(
        report_path=args.report,
        color=not args.no_color,
    )

    try:
        renderer.banner()

        if args.batch:
            if not args.batch.is_dir():
                renderer._emit(f"Error: not a directory: {args.batch}\n")
                return 2
            runner = BatchRunner(
                renderer,
                verbose=args.verbose,
                no_simulation=not args.simulation,
            )
            return runner.run(args.batch)

        else:
            path = args.circuit
            if not path.exists():
                renderer._emit(f"Error: file not found: {path}\n")
                return 2
            lang = args.lang or _detect_lang(path)
            runner = PipelineRunner(
                renderer,
                verbose=args.verbose,
                no_simulation=not args.simulation,
            )
            result = runner.run(path, lang)
            return 0 if result["passed"] else 1

    finally:
        renderer.close()


if __name__ == "__main__":
    sys.exit(main())
