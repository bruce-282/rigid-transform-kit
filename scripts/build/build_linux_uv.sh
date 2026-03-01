#!/bin/bash

# rigid-transform-kit Build Script for Linux with UV
#
# Usage: ./scripts/build/build_linux_uv.sh [PythonVersion] [extras]
# Example: ./scripts/build/build_linux_uv.sh                # Python 3.10, base
#          ./scripts/build/build_linux_uv.sh 3.11            # Python 3.11, base
#          ./scripts/build/build_linux_uv.sh 3.11 viz        # Python 3.11, with viz
#          ./scripts/build/build_linux_uv.sh 3.10 viz,dev    # Python 3.10, viz + dev

PYTHON_VERSION=${1:-3.10}
EXTRAS=${2:-""}

echo "=== rigid-transform-kit Build Script (uv, Python $PYTHON_VERSION) ==="

if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run from the project root."
    return 1 2>/dev/null || exit 1
fi

if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV with pip..."
    if ! pip install uv; then
        echo "Error: Failed to install UV."
        echo "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        return 1 2>/dev/null || exit 1
    fi
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile
    source ~/.profile
    echo "UV installed successfully!"
fi

echo "Installing Python $PYTHON_VERSION with UV..."
if ! uv python install $PYTHON_VERSION; then
    echo "Error: Failed to install Python $PYTHON_VERSION."
    return 1 2>/dev/null || exit 1
fi

echo "Creating virtual environment..."
if ! uv venv; then
    echo "Error: Failed to create virtual environment."
    return 1 2>/dev/null || exit 1
fi

if [ ! -f ".venv/bin/activate" ]; then
    echo "Error: Virtual environment activation script not found."
    return 1 2>/dev/null || exit 1
fi

source .venv/bin/activate

if [ -n "$EXTRAS" ]; then
    echo "Installing rigid-transform-kit with extras [$EXTRAS]..."
    uv pip install -e ".[$EXTRAS]"
else
    echo "Installing rigid-transform-kit (base)..."
    uv pip install -e .
fi

echo "Build completed! Virtual environment is ready."
echo "Activate with: source .venv/bin/activate"
