#Requires -Version 5.1
<#
.SYNOPSIS
    NeurDB Installer for Windows (Docker mode)

.DESCRIPTION
    Pulls and runs the NeurDB Docker image on Windows via Docker Desktop/WSL2.
    GPU is not supported on Windows — CPU image only.

.PARAMETER Version
    NeurDB version to install (default: latest)

.PARAMETER Port
    Host port for PostgreSQL (default: 5432)

.PARAMETER DataDir
    Optional bind mount for persistent data outside the container

.PARAMETER Uninstall
    Stop and remove the NeurDB container and image
#>

param(
    [string]$Version = "latest",
    [int]$Port = 5432,
    [string]$DataDir = "",
    [switch]$Uninstall,
    [switch]$Help
)

$ContainerName = "neurdb"
$Registry = "ghcr.io/neurdb/neurdb"
$Variant = "cpu"  # Windows: CPU only

function Show-Usage {
    Write-Host "NeurDB Installer for Windows"
    Write-Host ""
    Write-Host "Usage: .\install.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Version VERSION   NeurDB version to install (default: latest)"
    Write-Host "  -Port PORT         Host port for PostgreSQL (default: 5432)"
    Write-Host "  -DataDir PATH      Bind mount for persistent data"
    Write-Host "  -Uninstall         Stop and remove the NeurDB container and image"
    Write-Host "  -Help              Show this help message"
    exit 0
}

if ($Help) { Show-Usage }

# --- Uninstall mode ---
if ($Uninstall) {
    Write-Host "Stopping and removing NeurDB..."
    docker stop $ContainerName 2>$null
    docker rm -f $ContainerName 2>$null
    docker images --format "{{.Repository}}:{{.Tag}}" | Select-String "neurdb" | ForEach-Object {
        docker rmi $_.ToString() 2>$null
    }
    Write-Host "NeurDB has been uninstalled."
    exit 0
}

# --- Check prerequisites ---

# Check Docker Desktop
$dockerPath = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerPath) {
    Write-Host "Error: Docker is not installed." -ForegroundColor Red
    Write-Host "Download Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
}

# Check Docker daemon
try {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Write-Host "Error: Docker daemon is not running." -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again."
    exit 1
}

# Check WSL2 backend
$wslCheck = docker info 2>$null | Select-String "OSType: linux"
if (-not $wslCheck) {
    Write-Host "Warning: Docker does not appear to be using Linux containers." -ForegroundColor Yellow
    Write-Host "Ensure Docker Desktop is configured to use WSL2 backend with Linux containers."
}

# --- Determine image tag ---
if ($Version -eq "latest") {
    $ImageTag = "latest-$Variant"
} else {
    $ImageTag = "$Version-$Variant"
}
$Image = "${Registry}:${ImageTag}"

Write-Host "Pulling NeurDB image: $Image"
docker pull $Image
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to pull image." -ForegroundColor Red
    exit 1
}

# --- Stop existing container ---
docker stop $ContainerName 2>$null
docker rm -f $ContainerName 2>$null

# --- Build run command ---
$runArgs = @(
    "run", "-d",
    "--name", $ContainerName,
    "-p", "${Port}:5432",
    "--restart", "unless-stopped"
)

if ($DataDir -ne "") {
    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    }
    $runArgs += @("-v", "${DataDir}:/var/lib/neurdb/data")
}

$runArgs += $Image

Write-Host "Starting NeurDB container..."
& docker @runArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to start container." -ForegroundColor Red
    exit 1
}

# --- Wait for readiness ---
Write-Host -NoNewline "Waiting for NeurDB to be ready "
$maxWait = 120
$elapsed = 0
while ($true) {
    $result = docker exec $ContainerName /opt/neurdb/bin/psql -h localhost -p 5432 -U neurdb -c '\q' 2>$null
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host -NoNewline "."
    Start-Sleep -Seconds 2
    $elapsed += 2
    if ($elapsed -ge $maxWait) {
        Write-Host ""
        Write-Host "Warning: NeurDB did not become ready within ${maxWait}s." -ForegroundColor Yellow
        Write-Host "Check logs with: docker logs $ContainerName"
        exit 1
    }
}
Write-Host " OK"

# --- Print connection info ---
Write-Host ""
Write-Host "============================================"
Write-Host "  NeurDB is running!"
Write-Host "============================================"
Write-Host ""
Write-Host "  Connect with:"
Write-Host "    psql -h localhost -p $Port -U neurdb -d neurdb"
Write-Host ""
Write-Host "  Container name: $ContainerName"
Write-Host "  View logs:      docker logs -f $ContainerName"
Write-Host "  Stop:           docker stop $ContainerName"
Write-Host "  Uninstall:      .\install.ps1 -Uninstall"
Write-Host ""
