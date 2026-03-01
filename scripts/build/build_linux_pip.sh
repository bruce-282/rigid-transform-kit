#!/bin/bash

# rigid-transform-kit Build Script for Linux (pip)
#
# Usage: ./scripts/build/build_linux_pip.sh [extras]
# Example: ./scripts/build/build_linux_pip.sh          # base only
#          ./scripts/build/build_linux_pip.sh viz       # with rerun visualization
#          ./scripts/build/build_linux_pip.sh dev       # with dev/test tools
#          ./scripts/build/build_linux_pip.sh viz,dev   # both

EXTRAS=${1:-""}

echo "=== rigid-transform-kit Build Script (pip) ==="

if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run from the project root."
    return 1 2>/dev/null || exit 1
fi

PYTHON_CMD="python3"
if ! command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_CMD="python"
    if ! command -v $PYTHON_CMD &> /dev/null; then
        echo "Error: Python not found."
        return 1 2>/dev/null || exit 1
    fi
fi

echo "Using Python: $($PYTHON_CMD --version)"

if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    echo "Error: pip not found."
    return 1 2>/dev/null || exit 1
fi

$PYTHON_CMD -m pip install --upgrade pip

if [ -n "$EXTRAS" ]; then
    echo "Installing rigid-transform-kit with extras [$EXTRAS]..."
    $PYTHON_CMD -m pip install -e ".[$EXTRAS]"
else
    echo "Installing rigid-transform-kit (base)..."
    $PYTHON_CMD -m pip install -e .
fi

echo "Build completed successfully!"
echo
echo "Quick test: $PYTHON_CMD -c \"from rigid_transform_kit import Frame; print('OK')\""
