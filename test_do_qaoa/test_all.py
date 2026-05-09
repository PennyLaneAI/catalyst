#!/usr/bin/env python3
"""
DO-QAOA — Master Test Runner
==============================
Runs all 5 milestone scripts (phases 1–5).

Test files:
    test_phase1.py  — Phase 1: 4-qubit MaxCut MLIR round-trip + figure
    test_phase2.py  — Phase 2: Pearson r > 0.999 on Erdős-Rényi graphs + figure
    test_phase3.py  — Phase 3: 10-node BA, m=2, shots ≤ 130k + figure
    test_phase4.py  — Phase 4: do_qaoa() API, 12-node BA, m=3, shots ≤ 230k + figure
    test_phase5.py  — Phase 5: acceptance criteria C1–C11 + benchmark figures (Tasks 1–5)

Run:
    python test_do_qaoa/test_all.py
    python test_do_qaoa/test_all.py -v
    python test_do_qaoa/test_all.py --skip-mlir  # skip phases needing quantum-opt
"""

import argparse
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Phase 1 and 2 require quantum-opt for their MLIR steps
MLIR_PHASES = {"test_phase1.py", "test_phase2.py"}


def run_script(name, verbose=False):
    """Run a milestone script as a subprocess. Returns (passed, output)."""
    path = os.path.join(HERE, name)
    result = subprocess.run(
        [sys.executable, path],
        capture_output=not verbose,
        text=True,
        cwd=ROOT,
    )
    output = "" if verbose else (result.stdout + result.stderr)
    return result.returncode == 0, output


def main():
    parser = argparse.ArgumentParser(description="DO-QAOA master test runner")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--skip-mlir", action="store_true", help="skip phases needing quantum-opt")
    args = parser.parse_args()

    print("=" * 70)
    print("DO-QAOA — Full Test Suite")
    print("=" * 70)

    results = []

    # ── Milestone scripts (phases 1–5) ────────────────────────────────────────
    for phase_file in [
        "test_phase1.py",
        "test_phase2.py",
        "test_phase3.py",
        "test_phase4.py",
        "test_phase5.py",
    ]:
        if args.skip_mlir and phase_file in MLIR_PHASES:
            print(f"\n[SKIP] {phase_file}  (--skip-mlir)")
            results.append((phase_file, None))
            continue

        print(f"\n{'─'*70}")
        print(f"Running {phase_file} ...")
        print(f"{'─'*70}")

        passed, output = run_script(phase_file, verbose=args.verbose)
        if not args.verbose and output:
            print(output[-3000:] if len(output) > 3000 else output)

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"\n{phase_file}: {status}")
        results.append((phase_file, passed))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_passed = True
    for name, passed in results:
        if passed is None:
            print(f"  SKIP  {name}")
        elif passed:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}")
            all_passed = False

    print(f"{'='*70}")
    print("ALL PASS ✓" if all_passed else "SOME FAILED ✗")
    print(f"{'='*70}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
