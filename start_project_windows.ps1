param(
    [switch]$CleanLiveData,
    [switch]$GenerateSampleTraffic,
    [switch]$RunPipeline,
    [switch]$InstallPlaywrightBrowsers,
    [switch]$SkipDependencyInstall,
    [int]$Port = 8039
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [object[]]$ArgumentList = @()
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($ArgumentList -join ' ')"
    }
}

function Invoke-ProjectScript {
    param(
        [string]$RelativePath
    )
    $fullPath = Join-Path $ProjectRoot $RelativePath
    Invoke-Checked -FilePath $fullPath
}

function Invoke-ProjectPowerShellScript {
    param(
        [string]$RelativePath,
        [hashtable]$Parameters = @{}
    )
    $fullPath = Join-Path $ProjectRoot $RelativePath
    & $fullPath @Parameters
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $fullPath"
    }
}

$venvPath = Join-Path $ProjectRoot ".venv"
$activatePath = Join-Path $venvPath "Scripts\Activate.ps1"
$pythonInVenv = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $pythonInVenv)) {
    Write-Step "Creating Python virtual environment"
    Invoke-Checked -FilePath "python" -ArgumentList @("-m", "venv", ".venv")
}

if (-not (Test-Path $activatePath)) {
    throw "Virtual environment activation script not found at $activatePath"
}

Write-Step "Activating project virtual environment"
. $activatePath

if (-not $SkipDependencyInstall) {
    Write-Step "Installing project dependencies"
    Invoke-Checked -FilePath "python" -ArgumentList @("-m", "pip", "install", "-r", "requirements.txt")
    Invoke-Checked -FilePath "python" -ArgumentList @("-m", "pip", "install", "-e", ".")
    if ($InstallPlaywrightBrowsers) {
        Write-Step "Installing Playwright browser binaries"
        Invoke-Checked -FilePath "playwright" -ArgumentList @("install")
    }
}

Write-Step "Preparing local Nginx lab"
Invoke-ProjectPowerShellScript -RelativePath "lab\scripts\setup_nginx_windows.ps1" -Parameters @{ Port = $Port }

Write-Step "Stopping any previous repo-local Nginx instance"
Invoke-ProjectPowerShellScript -RelativePath "lab\scripts\stop_nginx_windows.ps1" -Parameters @{ Port = $Port }

if ($CleanLiveData) {
    Write-Step "Resetting live logs and prepared dataset folders"
    Invoke-ProjectScript -RelativePath "lab\scripts\reset_live_logs.bat"
    Remove-Item (Join-Path $ProjectRoot "data\live_labels\manual_labels.csv") -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $ProjectRoot "data\prepared_live") -Recurse -Force -ErrorAction SilentlyContinue
} elseif (-not (Test-Path (Join-Path $ProjectRoot "data\live_logs\access.log"))) {
    Write-Step "Creating clean live logs"
    Invoke-ProjectScript -RelativePath "lab\scripts\reset_live_logs.bat"
}

Write-Step "Starting local Nginx lab on port $Port"
Invoke-ProjectPowerShellScript -RelativePath "lab\scripts\start_nginx_windows.ps1" -Parameters @{ Port = $Port }

if ($GenerateSampleTraffic) {
    Write-Step "Generating sample human and bot traffic"
    $trafficScripts = @(
        "lab\scripts\generate_human_traffic.bat",
        "lab\scripts\generate_bot_bfs_traffic.bat",
        "lab\scripts\generate_bot_dfs_traffic.bat",
        "lab\scripts\generate_bot_linear_traffic.bat",
        "lab\scripts\generate_bot_stealth_traffic.bat",
        "lab\scripts\generate_bot_products_traffic.bat",
        "lab\scripts\generate_bot_articles_traffic.bat",
        "lab\scripts\generate_bot_revisit_traffic.bat",
        "lab\scripts\generate_bot_browser_hybrid_traffic.bat",
        "lab\scripts\generate_bot_browser_noise_traffic.bat",
        "lab\scripts\generate_bot_playwright_traffic.bat",
        "lab\scripts\generate_bot_selenium_traffic.bat"
    )
    foreach ($script in $trafficScripts) {
        Invoke-ProjectScript -RelativePath $script
    }
}

if ($RunPipeline) {
    Write-Step "Preparing live dataset"
    Invoke-ProjectScript -RelativePath "lab\scripts\prepare_live_dataset.bat"

    Write-Step "Exporting annotation template"
    Invoke-Checked -FilePath "python" -ArgumentList @(
        "-m", "wsd.export_label_template",
        "--session-summary", "data\prepared_live\session_summary.csv",
        "--output-path", "data\prepared_live\annotation_template.csv"
    )

    Write-Step "Running training and thesis experiment pipeline"
    Invoke-ProjectScript -RelativePath "lab\scripts\run_research_pipeline.bat"
}

Write-Host ""
Write-Host "Project is ready." -ForegroundColor Green
Write-Host "Website: http://127.0.0.1:$Port/"
Write-Host "Access log: data\live_logs\access.log"
Write-Host ""
Write-Host "Examples:"
Write-Host "  .\start_project_windows.ps1"
Write-Host "  .\start_project_windows.ps1 -CleanLiveData -GenerateSampleTraffic"
Write-Host "  .\start_project_windows.ps1 -CleanLiveData -GenerateSampleTraffic -RunPipeline"
