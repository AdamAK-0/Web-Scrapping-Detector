param(
    [int]$Port = 8039
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path

& (Join-Path $ScriptDir "setup_nginx_windows.ps1") -Port $Port

$nginxHome = Get-ChildItem -Directory -Path (Join-Path $ProjectRoot "tools") -Filter "nginx-*" | Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName
if (-not $nginxHome) {
    throw "Nginx directory not found under $ProjectRoot\tools"
}

$nginxExe = Join-Path $nginxHome "nginx.exe"
$confRel = "conf\nginx.conf"
$pidPath = Join-Path $nginxHome "logs\nginx.pid"
$errorLog = Join-Path $nginxHome "logs\error.log"

function Get-NginxPid {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $null
    }
    $raw = (Get-Content $Path -ErrorAction SilentlyContinue | Select-Object -First 1)
    if (-not $raw) {
        return $null
    }
    $raw = $raw.Trim()
    if ($raw -match '^\d+$') {
        return [int]$raw
    }
    return $null
}

function Test-Listening {
    param([int]$ListenPort)
    return @(
        netstat -ano |
        Select-String "LISTENING\s+\d+$" |
        Where-Object { $_.ToString() -match (":$ListenPort\s+") }
    ).Count -gt 0
}

function Get-RepoNginxPids {
    return @(
        Get-Process nginx -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $nginxExe } |
        Select-Object -ExpandProperty Id
    )
}

& $nginxExe -t -p "$nginxHome\\" -c $confRel
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ((Test-Listening -ListenPort $Port) -or @(Get-RepoNginxPids).Count -gt 0) {
    & (Join-Path $ScriptDir "stop_nginx_windows.ps1") -Port $Port
    if ($LASTEXITCODE -ne 0) {
        Write-Error "An existing nginx listener on port $Port could not be stopped cleanly."
        exit $LASTEXITCODE
    }
}

if (Test-Path $pidPath) {
    Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
}

Start-Process -FilePath $nginxExe -WorkingDirectory $nginxHome -ArgumentList @("-p", "$nginxHome\\", "-c", $confRel) -WindowStyle Hidden | Out-Null

$deadline = (Get-Date).AddSeconds(6)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    $startedPid = Get-NginxPid -Path $pidPath
    if (-not $startedPid) {
        continue
    }
    try {
        Get-Process -Id $startedPid -ErrorAction Stop | Out-Null
        if (Test-Listening -ListenPort $Port) {
            Write-Host "Nginx started with project config on http://127.0.0.1:$Port/"
            Write-Host "Access log: $(Join-Path $ProjectRoot 'data\live_logs\access.log')"
            exit 0
        }
    } catch {
        continue
    }
}

Write-Error "Nginx did not start cleanly on port $Port. Check $errorLog for details."
exit 1
