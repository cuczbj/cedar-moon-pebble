param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PipelineArgs)
$ErrorActionPreference = 'Stop'
if ($env:Q1_PYTHON) {
    $taskPython = $env:Q1_PYTHON
} elseif (Test-Path (Join-Path $PSScriptRoot '.venv/Scripts/python.exe')) {
    $taskPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
} elseif (Test-Path (Join-Path $PSScriptRoot '../.venv-q1/Scripts/python.exe')) {
    $taskPython = Join-Path $PSScriptRoot '../.venv-q1/Scripts/python.exe'
} else {
    $taskPython = 'python'
}
& $taskPython (Join-Path $PSScriptRoot 'run.py') @PipelineArgs
exit $LASTEXITCODE
