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

function Test-NginxInstallComplete {
    param(
        [string]$NginxRoot
    )

    $requiredPaths = @(
        (Join-Path $NginxRoot "nginx.exe"),
        (Join-Path $NginxRoot "conf"),
        (Join-Path $NginxRoot "conf\mime.types")
    )

    foreach ($requiredPath in $requiredPaths) {
        if (-not (Test-Path $requiredPath)) {
            return $false
        }
    }

    return $true
}

function Stop-RepoNginxProcesses {
    param(
        [string]$NginxRoot
    )

    $nginxExePath = (Join-Path $NginxRoot "nginx.exe")
    $repoPids = @(
        Get-Process nginx -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -eq $nginxExePath } |
        Select-Object -ExpandProperty Id
    )

    foreach ($procId in ($repoPids | Sort-Object -Unique)) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
        } catch {
            Write-Warning "Could not stop repo nginx PID ${procId}: $($_.Exception.Message)"
        }
    }
}

function Ensure-NginxZip {
    param(
        [string]$ArchivePath,
        [string]$Version
    )

    if (-not (Test-Path $ArchivePath)) {
        $DownloadUrl = "https://nginx.org/download/nginx-" + $Version + ".zip"
        Write-Host "Downloading Nginx from $DownloadUrl"
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $ArchivePath
    }
}

function Repair-NginxInstall {
    param(
        [string]$ArchivePath,
        [string]$DestinationRoot,
        [string]$Version
    )

    Ensure-NginxZip -ArchivePath $ArchivePath -Version $Version
    New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null

    $tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("nginx-repair-" + [guid]::NewGuid().ToString("N"))
    try {
        Expand-Archive -Path $ArchivePath -DestinationPath $tempRoot -Force
        $expandedHome = Join-Path $tempRoot ("nginx-" + $Version)
        if (-not (Test-Path $expandedHome)) {
            throw "Expanded Nginx archive did not contain expected folder: $expandedHome"
        }

        foreach ($requiredDir in @("conf")) {
            $sourceDir = Join-Path $expandedHome $requiredDir
            if (Test-Path $sourceDir) {
                Copy-Item $sourceDir -Destination $DestinationRoot -Recurse -Force
            }
        }

        foreach ($requiredFile in @("nginx.exe")) {
            $sourceFile = Join-Path $expandedHome $requiredFile
            $destFile = Join-Path $DestinationRoot $requiredFile
            if ((-not (Test-Path $destFile)) -and (Test-Path $sourceFile)) {
                Copy-Item $sourceFile -Destination $DestinationRoot -Force
            }
        }
    } finally {
        if (Test-Path $tempRoot) {
            Remove-Item $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

if ((Test-Path $NginxHome) -and -not (Test-NginxInstallComplete -NginxRoot $NginxHome)) {
    Write-Warning "Existing Nginx folder is incomplete. Repairing local Nginx install at $NginxHome"
    Stop-RepoNginxProcesses -NginxRoot $NginxHome
    Repair-NginxInstall -ArchivePath $ZipPath -DestinationRoot $NginxHome -Version $NginxVersion
}

if (-not (Test-Path $NginxHome)) {
    Ensure-NginxZip -ArchivePath $ZipPath -Version $NginxVersion
    Expand-Archive -Path $ZipPath -DestinationPath $ToolsDir -Force
}

if (-not (Test-NginxInstallComplete -NginxRoot $NginxHome)) {
    throw "Nginx install is still incomplete after extraction: $NginxHome"
}

$NginxLogs = Join-Path $NginxHome "logs"
New-Item -ItemType Directory -Force -Path $NginxLogs | Out-Null
$NginxTemp = Join-Path $NginxHome "temp"
New-Item -ItemType Directory -Force -Path $NginxTemp | Out-Null
foreach ($subdir in @("client_body_temp", "fastcgi_temp", "proxy_temp", "scgi_temp", "uwsgi_temp")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $NginxTemp $subdir) | Out-Null
}
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
