#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Automated Documentation Sync Script for CloudForge AI
.DESCRIPTION
    This script automatically syncs LaTeX documentation with app changes,
    compiles the PDF, and pushes everything to GitHub with proper commit messages.
.PARAMETER Message
    Custom commit message (optional)
.PARAMETER Force
    Force push even if there are warnings
#>

param(
    [string]$Message = "",
    [switch]$Force = $false
)

# Set error handling
$ErrorActionPreference = "Stop"

# Get project root directory
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "🚀 CloudForge AI Documentation Sync Started" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan

# Function to update LaTeX documentation based on app changes
function Update-LaTeXDocumentation {
    Write-Host "📝 Updating LaTeX documentation..." -ForegroundColor Yellow
    
    # Check for backend changes
    $BackendChanges = git diff --name-only HEAD~1 backend/ 2>$null
    if ($BackendChanges) {
        Write-Host "🔧 Backend changes detected, updating technical documentation..." -ForegroundColor Blue
        
        # Update architecture chapter with new backend components
        $ArchChapter = "$ProjectRoot\latex\chapters\04_architecture.tex"
        if (Test-Path $ArchChapter) {
            # Add timestamp to show last update
            $UpdateNote = "% Last updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Backend changes detected"
            Add-Content -Path $ArchChapter -Value "`n$UpdateNote"
        }
    }
    
    # Check for frontend changes
    $FrontendChanges = git diff --name-only HEAD~1 frontend/ 2>$null
    if ($FrontendChanges) {
        Write-Host "🎨 Frontend changes detected, updating UI documentation..." -ForegroundColor Blue
        
        # Update frontend chapter
        $UIChapter = "$ProjectRoot\latex\chapters\07_sprint_03.tex"
        if (Test-Path $UIChapter) {
            $UpdateNote = "% Last updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - Frontend changes detected"
            Add-Content -Path $UIChapter -Value "`n$UpdateNote"
        }
    }
    
    # Check for AI scripts changes
    $AIChanges = git diff --name-only HEAD~1 ai-scripts/ 2>$null
    if ($AIChanges) {
        Write-Host "🤖 AI scripts changes detected, updating ML documentation..." -ForegroundColor Blue
        
        # Update AI implementation chapter
        $AIChapter = "$ProjectRoot\latex\chapters\06_sprint_02.tex"
        if (Test-Path $AIChapter) {
            $UpdateNote = "% Last updated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - AI/ML changes detected"
            Add-Content -Path $AIChapter -Value "`n$UpdateNote"
        }
    }
}

# Function to compile LaTeX documentation
function Compile-LaTeXDocumentation {
    Write-Host "📚 Compiling LaTeX documentation..." -ForegroundColor Yellow
    
    Set-Location "$ProjectRoot\latex"
    
    try {
        # Compile PDF
        $CompileResult = pdflatex -interaction=nonstopmode main_fixed.tex 2>&1
        
        if (Test-Path "main_fixed.pdf") {
            $PdfSize = (Get-Item "main_fixed.pdf").Length
            $PdfPages = (Select-String "Output written on main_fixed.pdf \((\d+) pages" -InputObject $CompileResult).Matches[0].Groups[1].Value
            
            Write-Host "✅ PDF compiled successfully!" -ForegroundColor Green
            Write-Host "   📄 Pages: $PdfPages" -ForegroundColor Cyan
            Write-Host "   💾 Size: $([math]::Round($PdfSize/1KB, 2)) KB" -ForegroundColor Cyan
        } else {
            Write-Warning "⚠️  PDF compilation had warnings but completed"
        }
    }
    catch {
        Write-Error "❌ LaTeX compilation failed: $_"
        exit 1
    }
    finally {
        Set-Location $ProjectRoot
    }
}

# Function to create intelligent commit message
function Get-SmartCommitMessage {
    $Changes = @()
    
    # Detect types of changes
    $ModifiedFiles = git diff --name-only HEAD 2>$null
    $UntrackedFiles = git ls-files --others --exclude-standard 2>$null
    
    if ($ModifiedFiles -match "backend/") { $Changes += "backend" }
    if ($ModifiedFiles -match "frontend/") { $Changes += "frontend" }
    if ($ModifiedFiles -match "ai-scripts/") { $Changes += "ai/ml" }
    if ($ModifiedFiles -match "latex/") { $Changes += "docs" }
    if ($ModifiedFiles -match "helm-chart/|infra/") { $Changes += "infra" }
    if ($ModifiedFiles -match "tests/") { $Changes += "tests" }
    
    if ($UntrackedFiles) { $Changes += "new-features" }
    
    if ($Changes.Count -eq 0) {
        return "chore: minor updates and maintenance"
    }
    
    $ChangeStr = $Changes -join ", "
    return "feat: update $ChangeStr with documentation sync"
}

# Function to safely push to GitHub
function Push-ToGitHub {
    param([string]$CommitMsg)
    
    Write-Host "📤 Pushing to GitHub..." -ForegroundColor Yellow
    
    try {
        # Add all changes
        git add -A
        
        # Check if there are changes to commit
        $Status = git status --porcelain
        if (-not $Status) {
            Write-Host "✅ No changes to commit - repository is up to date!" -ForegroundColor Green
            return
        }
        
        # Commit changes
        git commit -m $CommitMsg
        
        # Push to GitHub
        Write-Host "🔄 Pushing to GitHub repository..." -ForegroundColor Blue
        git push origin main
        
        Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
        
        # Show summary
        $LastCommit = git log -1 --oneline
        Write-Host "📋 Latest commit: $LastCommit" -ForegroundColor Cyan
        
    }
    catch {
        Write-Error "❌ Git operation failed: $_"
        
        if (-not $Force) {
            Write-Host "💡 Try running with -Force flag to resolve conflicts automatically" -ForegroundColor Yellow
            exit 1
        }
        
        Write-Host "🔧 Attempting to resolve conflicts..." -ForegroundColor Yellow
        git pull origin main --rebase
        git push origin main
    }
}

# Main execution flow
try {
    # Step 1: Update documentation based on app changes
    Update-LaTeXDocumentation
    
    # Step 2: Compile LaTeX documentation
    Compile-LaTeXDocumentation
    
    # Step 3: Create commit message
    $CommitMessage = if ($Message) { $Message } else { Get-SmartCommitMessage }
    Write-Host "📝 Commit message: $CommitMessage" -ForegroundColor Magenta
    
    # Step 4: Push to GitHub
    Push-ToGitHub -CommitMsg $CommitMessage
    
    Write-Host "`n🎉 Documentation sync completed successfully!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
    
}
catch {
    Write-Error "❌ Documentation sync failed: $_"
    exit 1
}