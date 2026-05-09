#!/usr/bin/env python3
"""
DO-QAOA — Master Test Runner
==============================
Runs all 4 milestone scripts and the full acceptance criteria suite (C1–C11).

Test files:
    test_phase1.py  — Phase 1: 4-qubit MaxCut MLIR round-trip (milestone_maxcut)
    test_phase2.py  — Phase 2: Pearson r > 0.999 on Erdős-Rényi graphs
    test_phase3.py  — Phase 3: 10-node BA, m=2, shots ≤ 130k
    test_phase4.py  — Phase 4: do_qaoa() API, 12-node BA, m=3, shots ≤ 230k
    test_acceptance_criteria.py — Acceptance criteria C1–C11 (paper benchmarks)

Run:
    python test_do_qaoa/test_all.py
    python test_do_qaoa/test_all.py -v
    python test_do_qaoa/test_all.py --fast    # skip slow acceptance tests (C3,C4,C5,C10)
    python test_do_qaoa/test_all.py --skip-mlir  # skip tests needing quantum-opt
"""

import argparse
import subprocess
import sys
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SLOW_TESTS = {"TestC3_PowerLawARG", "TestC4_ShotBudget", "TestC5_ERGraphARG", "TestC10_WallClockSpeedup"}

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


def run_acceptance(fast=False, verbose=False):
    """Run acceptance criteria C1–C11 via unittest. Returns (passed, n_run, n_fail)."""
    sys.path.insert(0, HERE)
    import test_acceptance_criteria as _tac

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for cls_name in [
        "TestC1_PearsonNoCoeff", "TestC2_PearsonWithCoeff", "TestC3_PowerLawARG",
        "TestC4_ShotBudget", "TestC5_ERGraphARG", "TestC6_FrozenQubitsReference",
        "TestC7_WithinGraphCorrelation", "TestC8_CNOTCount", "TestC9_LandscapeOverlap",
        "TestC10_WallClockSpeedup", "TestC11_PlatformAndImports",
    ]:
        if fast and cls_name in SLOW_TESTS:
            continue
        cls = getattr(_tac, cls_name)
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1, stream=sys.stdout)
    result = runner.run(suite)
    failures = len(result.failures) + len(result.errors)
    return result.wasSuccessful(), result.testsRun, failures


def main():
    parser = argparse.ArgumentParser(description="DO-QAOA master test runner")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true", help="skip slow acceptance tests")
    parser.add_argument("--skip-mlir", action="store_true", help="skip phases needing quantum-opt")
    args = parser.parse_args()

    print("=" * 70)
    print("DO-QAOA — Full Test Suite")
    print("=" * 70)

    results = []

    # ── Milestone scripts (phases 1-4) ────────────────────────────────────────
    for phase_file in ["test_phase1.py", "test_phase2.py", "test_phase3.py", "test_phase4.py"]:
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

    # ── Acceptance criteria C1–C11 ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Running test_acceptance_criteria.py (C1–C11) ...")
    if args.fast:
        print("  (fast mode: C3, C4, C5, C10 skipped)")
    print(f"{'─'*70}")

    ac_passed, n_run, n_fail = run_acceptance(fast=args.fast, verbose=args.verbose)
    status = "PASS ✓" if ac_passed else "FAIL ✗"
    print(f"\ntest_acceptance_criteria.py ({n_run} tests, {n_fail} failures): {status}")
    results.append(("test_acceptance_criteria.py", ac_passed))

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
