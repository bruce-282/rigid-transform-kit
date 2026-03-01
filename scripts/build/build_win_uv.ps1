# rigid-transform-kit Build Script for Windows with UV
#
# Usage: .\scripts\build\build_win_uv.ps1 [PythonVersion] [extras]
# Example: .\scripts\build\build_win_uv.ps1                  # Python 3.10, base
#          .\scripts\build\build_win_uv.ps1 3.11              # Python 3.11, base
#          .\scripts\build\build_win_uv.ps1 3.11 viz          # Python 3.11, with viz
#          .\scripts\build\build_win_uv.ps1 3.10 "viz,dev"    # Python 3.10, viz + dev

param(
    [string]$PythonVersion = "3.10",
    [string]$Extras = ""
)

Write-Host "=== rigid-transform-kit Build Script (uv, Python $PythonVersion) ==="

if (-not (Test-Path "pyproject.toml")) {
    Write-Host "Error: pyproject.toml not found. Please run from the project root."
    exit 1
}

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "UV is not installed. Installing UV with pip..."
    try {
        pip install uv
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error: Failed to install UV."
            Write-Host "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
            exit 1
        }
        Write-Host "UV installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "Error: Failed to install UV. Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }
}

Write-Host "Installing Python $PythonVersion with UV..."
uv python install $PythonVersion
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to install Python $PythonVersion."
    exit 1
}

Write-Host "Creating virtual environment..."
uv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to create virtual environment."
    exit 1
}

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Error: Virtual environment activation script not found."
    exit 1
}

& ".venv\Scripts\Activate.ps1"

if ($Extras -ne "") {
    Write-Host "Installing rigid-transform-kit with extras [$Extras]..."
    uv pip install -e ".[$Extras]"
} else {
    Write-Host "Installing rigid-transform-kit (base)..."
    uv pip install -e .
}

Write-Host "Build completed! Virtual environment is ready." -ForegroundColor Green
Write-Host "Activate with: .venv\Scripts\activate"
