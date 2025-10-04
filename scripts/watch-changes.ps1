#!/usr/bin/env pwsh
<#
.SYNOPSIS
    File Watcher for CloudForge AI Documentation
.DESCRIPTION
    Monitors app files for changes and automatically syncs documentation
#>

$ErrorActionPreference = "Stop"

Write-Host "👀 Starting CloudForge AI Documentation Watcher..." -ForegroundColor Green
Write-Host "Monitoring for changes in: backend/, frontend/, ai-scripts/" -ForegroundColor Cyan

# Get project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Directories to watch
$WatchPaths = @(
    "$ProjectRoot\backend",
    "$ProjectRoot\frontend", 
    "$ProjectRoot\ai-scripts",
    "$ProjectRoot\helm-chart",
    "$ProjectRoot\infra"
)

# Create file system watchers
$Watchers = @()

foreach ($Path in $WatchPaths) {
    if (Test-Path $Path) {
        $Watcher = New-Object System.IO.FileSystemWatcher
        $Watcher.Path = $Path
        $Watcher.IncludeSubdirectories = $true
        $Watcher.EnableRaisingEvents = $true
        
        # Define the action to take when a file changes
        $Action = {
            $ChangedFile = $Event.SourceEventArgs.FullPath
            $ChangeType = $Event.SourceEventArgs.ChangeType
            
            # Ignore temporary files and node_modules
            if ($ChangedFile -match "node_modules|\.tmp|\.log|\.swp") { return }
            
            Write-Host "`n🔄 Change detected: $ChangeType - $ChangedFile" -ForegroundColor Yellow
            
            # Wait a moment for file operations to complete
            Start-Sleep -Seconds 2
            
            try {
                # Run the sync script
                Write-Host "📝 Syncing documentation..." -ForegroundColor Blue
                & "$ProjectRoot\scripts\sync-docs.ps1" -Message "auto: sync docs after $ChangeType in $(Split-Path -Leaf $ChangedFile)"
            }
            catch {
                Write-Warning "⚠️  Auto-sync failed: $_"
            }
        }
        
        # Register event handlers
        Register-ObjectEvent -InputObject $Watcher -EventName "Changed" -Action $Action
        Register-ObjectEvent -InputObject $Watcher -EventName "Created" -Action $Action
        Register-ObjectEvent -InputObject $Watcher -EventName "Deleted" -Action $Action
        
        $Watchers += $Watcher
        Write-Host "✅ Watching: $Path" -ForegroundColor Green
    }
}

Write-Host "`n🚀 Documentation watcher is now active!" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop watching..." -ForegroundColor Yellow

try {
    # Keep the script running
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
finally {
    # Cleanup watchers
    foreach ($Watcher in $Watchers) {
        $Watcher.EnableRaisingEvents = $false
        $Watcher.Dispose()
    }
    Write-Host "`n👋 Documentation watcher stopped." -ForegroundColor Red
}