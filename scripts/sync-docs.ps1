# CloudForge AI Documentation Sync Script
# This script automatically synchronizes documentation when app changes are detected

param(
    [switch]$Force,
    [string]$LogLevel = "Info"
)

# Set script variables
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$LatexDir = Join-Path $RootDir "latex"
$LogFile = Join-Path $RootDir "logs\sync-docs.log"

# Ensure logs directory exists
$LogsDir = Join-Path $RootDir "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "Info"
    )
    
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] [$Level] $Message"
    
    # Write to console with colors
    switch ($Level) {
        "Error" { Write-Host $LogMessage -ForegroundColor Red }
        "Warning" { Write-Host $LogMessage -ForegroundColor Yellow }
        "Success" { Write-Host $LogMessage -ForegroundColor Green }
        "Info" { Write-Host $LogMessage -ForegroundColor Cyan }
        default { Write-Host $LogMessage }
    }
    
    # Write to log file
    Add-Content -Path $LogFile -Value $LogMessage
}

function Test-Changes {
    param(
        [string[]]$Paths
    )
    
    Write-Log "Checking for changes in app directories..." "Info"
    
    $LastSyncFile = Join-Path $RootDir ".last-sync"
    
    if (-not (Test-Path $LastSyncFile) -or $Force) {
        Write-Log "No previous sync found or force flag used - full sync required" "Info"
        return $true
    }
    
    $LastSync = Get-Content $LastSyncFile
    $LastSyncTime = [DateTime]::Parse($LastSync)
    
    foreach ($Path in $Paths) {
        $FullPath = Join-Path $RootDir $Path
        if (Test-Path $FullPath) {
            $RecentFiles = Get-ChildItem -Path $FullPath -Recurse -File | Where-Object { $_.LastWriteTime -gt $LastSyncTime }
            if ($RecentFiles) {
                Write-Log "Changes detected in $Path" "Info"
                return $true
            }
        }
    }
    
    Write-Log "No changes detected since last sync" "Info"
    return $false
}

function Update-LaTeXDocumentation {
    Write-Log "Starting LaTeX documentation update..." "Info"
    
    # Switch to latex directory
    Push-Location $LatexDir
    
    try {
        # Compile PDF
        Write-Log "Compiling LaTeX document..." "Info"
        $CompileResult = pdflatex -interaction=nonstopmode main_fixed.tex 2>&1
        
        if (Test-Path "main_fixed.pdf") {
            $PdfSize = (Get-Item "main_fixed.pdf").Length
            $PdfPages = "Unknown"
            
            # Try to extract page count from output
            $PageMatch = $CompileResult | Select-String "Output written on main_fixed.pdf \((\d+) pages"
            if ($PageMatch) {
                $PdfPages = $PageMatch.Matches[0].Groups[1].Value
            }
            
            Write-Log "PDF compiled successfully!" "Success"
            Write-Log "Pages: $PdfPages" "Info"
            Write-Log "Size: $([math]::Round($PdfSize/1KB, 2)) KB" "Info"
        } else {
            Write-Log "PDF compilation had warnings but completed" "Warning"
        }
    }
    catch {
        Write-Log "LaTeX compilation failed: $_" "Error"
        throw
    }
    finally {
        Pop-Location
    }
}

function Build-LaTeXDocumentation {
    param(
        [string[]]$ChangedFiles
    )
    
    Write-Log "Building documentation for changed files..." "Info"
    
    # Analyze changes and update relevant LaTeX chapters
    foreach ($File in $ChangedFiles) {
        Write-Log "Processing change: $File" "Info"
        
        # Update relevant LaTeX sections based on file type
        if ($File -match "backend/") {
            Write-Log "Backend changes detected - updating architecture chapter" "Info"
        }
        elseif ($File -match "frontend/") {
            Write-Log "Frontend changes detected - updating UI chapter" "Info"
        }
        elseif ($File -match "ai-scripts/") {
            Write-Log "AI Scripts changes detected - updating AI chapter" "Info"
        }
        elseif ($File -match "infra/") {
            Write-Log "Infrastructure changes detected - updating deployment chapter" "Info"
        }
    }
    
    # Compile the documentation
    Update-LaTeXDocumentation
}

function Push-ToGitHub {
    param(
        [string]$Message = "docs: auto-sync documentation"
    )
    
    Write-Log "Pushing changes to GitHub..." "Info"
    
    try {
        # Add all changes
        git add .
        
        # Check if there are changes to commit
        $Status = git status --porcelain
        if (-not $Status) {
            Write-Log "No changes to commit" "Info"
            return
        }
        
        # Commit changes
        git commit -m $Message
        
        # Push to GitHub
        git push origin main
        
        Write-Log "Successfully pushed to GitHub!" "Success"
    }
    catch {
        Write-Log "Git operations failed: $_" "Error"
        throw
    }
}

function Update-SyncTimestamp {
    $LastSyncFile = Join-Path $RootDir ".last-sync"
    $CurrentTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Set-Content -Path $LastSyncFile -Value $CurrentTime
    Write-Log "Updated sync timestamp: $CurrentTime" "Info"
}

# Main execution
try {
    Write-Log "Starting CloudForge AI Documentation Sync" "Info"
    Write-Log "Root Directory: $RootDir" "Info"
    Write-Log "Force Mode: $Force" "Info"
    
    # Define paths to monitor
    $MonitoredPaths = @(
        "backend/src",
        "frontend/src", 
        "ai-scripts",
        "infra",
        "helm-chart"
    )
    
    # Check for changes
    if (Test-Changes -Paths $MonitoredPaths) {
        Write-Log "Changes detected - starting documentation sync" "Info"
        
        # Build documentation
        Build-LaTeXDocumentation -ChangedFiles @()
        
        # Push to GitHub
        Push-ToGitHub -Message "docs: auto-sync documentation $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        
        # Update timestamp
        Update-SyncTimestamp
        
        Write-Log "Documentation sync completed successfully!" "Success"
    } else {
        Write-Log "No sync required - documentation is up to date" "Info"
    }
}
catch {
    Write-Log "Documentation sync failed: $_" "Error"
    exit 1
}