param(
    [int]$Port = 8039,
    [string]$NginxVersion = "1.29.8"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$ToolsDir = Join-Path $ProjectRoot "tools"
$ZipPath = Join-Path $ToolsDir ("nginx-" + $NginxVersion + ".zip")
$NginxHome = Join-Path $ToolsDir ("nginx-" + $NginxVersion)
$LiveLogs = Join-Path $ProjectRoot "data\live_logs"
$LiveLabels = Join-Path $ProjectRoot "data\live_labels"
$GeneratedDir = Join-Path $ProjectRoot "lab\generated"

New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
New-Item -ItemType Directory -Force -Path $LiveLabels | Out-Null
New-Item -ItemType Directory -Force -Path $GeneratedDir | Out-Null

if (-not (Test-Path $NginxHome)) {
    if (-not (Test-Path $ZipPath)) {
        $DownloadUrl = "https://nginx.org/download/nginx-" + $NginxVersion + ".zip"
        Write-Host "Downloading Nginx from $DownloadUrl"
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath
    }
    Expand-Archive -Path $ZipPath -DestinationPath $ToolsDir -Force
}

$NginxLogs = Join-Path $NginxHome "logs"
New-Item -ItemType Directory -Force -Path $NginxLogs | Out-Null
$AccessLogPath = (Join-Path $NginxLogs "access.log").Replace('\', '/')
$ErrorLogPath = (Join-Path $NginxLogs "error.log").Replace('\', '/')

if (Test-Path $LiveLogs) {
    $existing = Get-Item $LiveLogs -Force
    $isCorrectJunction = $false
    if ($existing.Attributes.ToString().Contains('ReparsePoint') -and $existing.LinkType -eq 'Junction') {
        try {
            $target = ($existing.Target | Select-Object -First 1)
            if ($target) {
                $resolvedTarget = (Resolve-Path $target).Path
                $resolvedNginxLogs = (Resolve-Path $NginxLogs).Path
                if ($resolvedTarget -eq $resolvedNginxLogs) { $isCorrectJunction = $true }
            }
        } catch {}
    }
    if (-not $isCorrectJunction) {
        Remove-Item $LiveLogs -Recurse -Force
    }
}

if (-not (Test-Path $LiveLogs)) {
    try {
        New-Item -ItemType Junction -Path $LiveLogs -Target $NginxLogs | Out-Null
    } catch {
        Write-Warning "Could not create live_logs junction. Falling back to a normal directory: $($_.Exception.Message)"
        New-Item -ItemType Directory -Force -Path $LiveLogs | Out-Null
    }
}

Push-Location $ProjectRoot
python -m wsd.lab_setup --project-root . --port $Port --access-log $AccessLogPath --error-log $ErrorLogPath --write-conf (Join-Path $NginxHome "conf\nginx.conf") --write-report (Join-Path $GeneratedDir "website_link_check.json")
Pop-Location

Write-Host "Nginx home: $NginxHome"
Write-Host "Website: http://127.0.0.1:$Port/"
Write-Host "Nginx logs: $NginxLogs"
Write-Host "Live logs alias: $LiveLogs"
Write-Host "Access log: $(Join-Path $LiveLogs 'access.log')"
