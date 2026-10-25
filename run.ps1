param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArguments
)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workspaceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$sessionPython = Join-Path $workspaceRoot 'work\.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $projectPython) {
    $taskPython = $projectPython
} elseif (Test-Path -LiteralPath $sessionPython) {
    $taskPython = $sessionPython
} else {
    throw 'Create a Python environment using the README setup instructions first.'
}
Push-Location $PSScriptRoot
try {
    & $taskPython -m finance_rag.cli @CliArguments
    $taskExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $taskExitCode
