$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Create the Python virtual environment and install the project first. See README.md.'
}
& '.\.venv\Scripts\python.exe' -m prelabel serve
