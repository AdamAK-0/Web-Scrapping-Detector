param(
    [int]$Port = 8039
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$nginxHome = Get-ChildItem -Directory -Path (Join-Path $ProjectRoot "tools") -Filter "nginx-*" | Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName

if (-not $nginxHome) {
    throw "Nginx directory not found under $ProjectRoot\tools"
}

$nginxExe = Join-Path $nginxHome "nginx.exe"
$confRel = "conf\nginx.conf"
$pidPath = Join-Path $nginxHome "logs\nginx.pid"

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

function Get-ListeningNginxPids {
    param([int]$ListenPort)
    $pids = @(
        netstat -ano |
        Select-String "LISTENING\s+\d+$" |
        Where-Object { $_.ToString() -match (":$ListenPort\s+") } |
        ForEach-Object {
            if ($_.ToString() -match 'LISTENING\s+(\d+)$') {
                [int]$Matches[1]
            }
        } |
        Sort-Object -Unique
    )
    $nginxPids = @()
    foreach ($procId in $pids) {
        try {
            $proc = Get-Process -Id $procId -ErrorAction Stop
            if ($proc.ProcessName -eq "nginx") {
                $nginxPids += $procId
            }
        } catch {
            continue
        }
    }
    return $nginxPids
}

function Get-RepoNginxPids {
    return @(
        Get-Process nginx -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $nginxExe } |
        Select-Object -ExpandProperty Id
    )
}

function Test-Listening {
    param([int]$ListenPort)
    return @(Get-ListeningNginxPids -ListenPort $ListenPort).Count -gt 0
}

$existingPid = Get-NginxPid -Path $pidPath
if ($existingPid) {
    try {
        & $nginxExe -p "$nginxHome\\" -c $confRel -s stop | Out-Null
    } catch {
        Write-Warning "Signal stop failed for PID file process ${existingPid}: $($_.Exception.Message)"
    }
}

$deadline = (Get-Date).AddSeconds(6)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    if ((-not (Test-Listening -ListenPort $Port)) -and @(Get-RepoNginxPids).Count -eq 0) {
        if (Test-Path $pidPath) {
            Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Nginx stopped on port $Port."
        exit 0
    }
}

$fallbackPids = @($existingPid) + @(Get-ListeningNginxPids -ListenPort $Port) + @(Get-RepoNginxPids)
$fallbackPids = @($fallbackPids | Where-Object { $_ } | Sort-Object -Unique)
foreach ($procId in $fallbackPids) {
    try {
        Stop-Process -Id $procId -Force -ErrorAction Stop
    } catch {
        taskkill /F /PID $procId > $null 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Could not stop nginx PID ${procId}: $($_.Exception.Message)"
        }
    }
}

$deadline = (Get-Date).AddSeconds(4)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 250
    if ((-not (Test-Listening -ListenPort $Port)) -and @(Get-RepoNginxPids).Count -eq 0) {
        if (Test-Path $pidPath) {
            Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
        }
        Write-Host "Nginx stopped on port $Port."
        exit 0
    }
}

$remaining = @((Get-ListeningNginxPids -ListenPort $Port) + (Get-RepoNginxPids) | Sort-Object -Unique)
Write-Error "Nginx is still listening on port $Port. Remaining repo nginx PIDs: $($remaining -join ', ')"
exit 1
