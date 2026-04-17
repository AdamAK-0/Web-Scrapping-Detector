param(
    [string]$LogPath = "run_logs.txt",
    [switch]$CleanLiveData,
    [switch]$GenerateSampleTraffic,
    [switch]$RunPipeline,
    [switch]$InstallPlaywrightBrowsers,
    [switch]$SkipDependencyInstall,
    [switch]$UseGeneratedHumanTraffic,
    [int]$Port = 8039
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

function ConvertTo-CmdArgument {
    param([string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

$batPath = Join-Path $ScriptDir "run_research_pipeline.bat"
$startProjectPath = Join-Path $ProjectRoot "start_project_windows.ps1"
$workflowMode = $CleanLiveData -or $GenerateSampleTraffic -or $RunPipeline -or $InstallPlaywrightBrowsers -or $SkipDependencyInstall -or $UseGeneratedHumanTraffic

"=== Research pipeline started: $(Get-Date -Format s) ===" | Tee-Object -FilePath $resolvedLogPath
if ($workflowMode) {
    $startArgs = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $startProjectPath,
        "-Port", $Port
    )
    if ($CleanLiveData) { $startArgs += "-CleanLiveData" }
    if ($GenerateSampleTraffic) { $startArgs += "-GenerateSampleTraffic" }
    if ($RunPipeline) { $startArgs += "-RunPipeline" }
    if ($InstallPlaywrightBrowsers) { $startArgs += "-InstallPlaywrightBrowsers" }
    if ($SkipDependencyInstall) { $startArgs += "-SkipDependencyInstall" }
    if ($UseGeneratedHumanTraffic) { $startArgs += "-UseGeneratedHumanTraffic" }
    $commandLine = "powershell.exe " + (($startArgs | ForEach-Object { ConvertTo-CmdArgument $_ }) -join " ") + " 2>&1"
    & cmd.exe /d /c $commandLine | Tee-Object -FilePath $resolvedLogPath -Append
} else {
    & cmd.exe /c "`"$batPath`" 2>&1" | Tee-Object -FilePath $resolvedLogPath -Append
}
$exitCode = $LASTEXITCODE
"=== Research pipeline finished: $(Get-Date -Format s) | ExitCode=$exitCode ===" | Tee-Object -FilePath $resolvedLogPath -Append

if ($exitCode -ne 0) {
    throw "Research pipeline failed. See log: $resolvedLogPath"
}

Write-Host "Saved run log to $resolvedLogPath"
