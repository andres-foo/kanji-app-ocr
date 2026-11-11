[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Virtual environment not found. Create it and install requirements first; see README.md."
    exit 1
}

Set-Location -LiteralPath $Root
$env:PYTHONUNBUFFERED = "1"

Write-Host "Starting KanjiOCRVibe OCR on http://127.0.0.1:$Port"
Write-Host "Press Ctrl+C to stop."

& $Python -m uvicorn app:app --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
