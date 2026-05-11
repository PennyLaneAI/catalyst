#!/bin/bash
#
# Comprehensive test runner for QASM3 translation tests
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "QASM3 Translation Test Suite"
echo "======================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Setup environment
export PYTHONPATH="${ROOT_DIR}/mlir/llvm-project/build/tools/mlir/python_packages/mlir_core:${PYTHONPATH}"

echo -e "${YELLOW}Environment Setup:${NC}"
echo "  Root: $ROOT_DIR"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking Dependencies:${NC}"

if [ ! -f "${ROOT_DIR}/mlir/build/bin/quantum-opt" ]; then
    echo -e "${RED}✗ quantum-opt not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ quantum-opt found${NC}"

if [ ! -f "${ROOT_DIR}/mlir/build/bin/quantum-translate" ]; then
    echo -e "${RED}✗ quantum-translate not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ quantum-translate found${NC}"

if ! python -c "import qiskit" 2>/dev/null; then
    echo -e "${RED}✗ Qiskit not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Qiskit available${NC}"

if ! python -c "import qiskit_aer" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Qiskit Aer not available (semantic validation will be skipped)${NC}"
else
    echo -e "${GREEN}✓ Qiskit Aer available${NC}"
fi
echo ""

# Count circuits
CIRCUIT_COUNT=$(find "${SCRIPT_DIR}/qasm3_circuits" -name "*.qasm" | wc -l)
echo -e "${YELLOW}Test Circuits: ${CIRCUIT_COUNT}${NC}"
echo ""

# Run tests based on mode
MODE="${1:-all}"

case "$MODE" in
    "legacy")
        echo -e "${YELLOW}Running Legacy Test Suite...${NC}"
        cd "$ROOT_DIR"
        python test_qcc_feature/qasm3/test_translation.py
        ;;

    "pytest")
        echo -e "${YELLOW}Running Pytest Suite...${NC}"
        cd "$ROOT_DIR"

        if ! python -c "import pytest" 2>/dev/null; then
            echo -e "${RED}✗ pytest not installed${NC}"
            echo "Install with: pip install pytest pytest-xdist pytest-cov"
            exit 1
        fi

        pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py -v
        ;;

    "pytest-parallel")
        echo -e "${YELLOW}Running Pytest Suite (Parallel)...${NC}"
        cd "$ROOT_DIR"

        if ! python -c "import pytest" 2>/dev/null; then
            echo -e "${RED}✗ pytest not installed${NC}"
            exit 1
        fi

        if ! python -c "import xdist" 2>/dev/null; then
            echo -e "${YELLOW}⚠ pytest-xdist not available, running sequentially${NC}"
            pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py -v
        else
            pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py -v -n auto
        fi
        ;;

    "coverage")
        echo -e "${YELLOW}Running Tests with Coverage...${NC}"
        cd "$ROOT_DIR"

        if ! python -c "import pytest_cov" 2>/dev/null; then
            echo -e "${RED}✗ pytest-cov not installed${NC}"
            echo "Install with: pip install pytest-cov"
            exit 1
        fi

        pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py -v --cov=test_translation_qasm3 --cov-report=html --cov-report=term
        echo ""
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;

    "all"|*)
        echo -e "${YELLOW}Running All Test Suites...${NC}"
        echo ""

        echo "=== Legacy Test Suite ==="
        cd "$ROOT_DIR"
        python test_qcc_feature/qasm3/test_translation.py
        LEGACY_STATUS=$?
        echo ""

        if python -c "import pytest" 2>/dev/null; then
            echo "=== Pytest Suite ==="
            pytest test_qcc_feature/qasm3/test_qasm3_translation_pytest.py -v
            PYTEST_STATUS=$?
            echo ""

            if [ $LEGACY_STATUS -eq 0 ] && [ $PYTEST_STATUS -eq 0 ]; then
                echo -e "${GREEN}✓ All tests passed!${NC}"
                exit 0
            else
                echo -e "${RED}✗ Some tests failed${NC}"
                exit 1
            fi
        else
            if [ $LEGACY_STATUS -eq 0 ]; then
                echo -e "${GREEN}✓ Legacy tests passed!${NC}"
                echo -e "${YELLOW}⚠ Install pytest for additional tests${NC}"
                exit 0
            else
                echo -e "${RED}✗ Tests failed${NC}"
                exit 1
            fi
        fi
        ;;
esac

echo ""
echo "======================================"
echo "Test run complete"
echo "======================================"
