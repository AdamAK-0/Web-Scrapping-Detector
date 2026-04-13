param(
    [string]$LogPath = "run_logs.txt"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
Set-Location $ProjectRoot

$resolvedLogPath = if ([System.IO.Path]::IsPathRooted($LogPath)) {
    $LogPath
} else {
    Join-Path $ProjectRoot $LogPath
}

$logDir = Split-Path -Parent $resolvedLogPath
if ($logDir -and -not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$batPath = Join-Path $ScriptDir "run_research_pipeline.bat"

"=== Research pipeline started: $(Get-Date -Format s) ===" | Tee-Object -FilePath $resolvedLogPath
& cmd.exe /c "`"$batPath`" 2>&1" | Tee-Object -FilePath $resolvedLogPath -Append
$exitCode = $LASTEXITCODE
"=== Research pipeline finished: $(Get-Date -Format s) | ExitCode=$exitCode ===" | Tee-Object -FilePath $resolvedLogPath -Append

if ($exitCode -ne 0) {
    throw "Research pipeline failed. See log: $resolvedLogPath"
}

Write-Host "Saved run log to $resolvedLogPath"
