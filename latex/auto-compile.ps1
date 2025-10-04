#!/usr/bin/env powershell
# CloudForge AI - Automated LaTeX Setup and Compilation
# This script downloads and installs TinyTeX, then compiles the document

param(
    [switch]$SkipInstall,
    [switch]$CleanOnly
)

Write-Host "CloudForge AI - LaTeX Document Compilation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Function to check if LaTeX is installed
function Test-LaTeXInstallation {
    try {
        $pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
        if ($pdflatex) {
            Write-Host "✓ LaTeX installation found: $($pdflatex.Source)" -ForegroundColor Green
            return $true
        }
    } catch {}
    
    Write-Host "✗ LaTeX not found" -ForegroundColor Red
    return $false
}

# Function to install TinyTeX
function Install-TinyTeX {
    Write-Host "Installing TinyTeX (lightweight LaTeX distribution)..." -ForegroundColor Yellow
    
    try {
        # Download TinyTeX installer
        $url = "https://github.com/rstudio/tinytex-releases/releases/download/daily/TinyTeX-1.zip"
        $output = "$env:TEMP\TinyTeX.zip"
        
        Write-Host "Downloading TinyTeX..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
        
        # Extract to a temporary location
        $extractPath = "$env:TEMP\TinyTeX"
        if (Test-Path $extractPath) {
            Remove-Item $extractPath -Recurse -Force
        }
        
        Write-Host "Extracting TinyTeX..." -ForegroundColor Yellow
        Expand-Archive -Path $output -DestinationPath $extractPath -Force
        
        # Install TinyTeX
        $installer = Get-ChildItem -Path $extractPath -Name "install-windows.bat" -Recurse | Select-Object -First 1
        if ($installer) {
            $installerPath = Join-Path $extractPath $installer
            Write-Host "Running TinyTeX installer..." -ForegroundColor Yellow
            Start-Process -FilePath $installerPath -Wait -NoNewWindow
        } else {
            throw "TinyTeX installer not found"
        }
        
        # Refresh PATH
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
        
        Write-Host "✓ TinyTeX installation completed" -ForegroundColor Green
        return $true
        
    } catch {
        Write-Host "✗ Failed to install TinyTeX: $_" -ForegroundColor Red
        return $false
    }
}

# Function to compile LaTeX document
function Invoke-LaTeXCompilation {
    Write-Host "Starting LaTeX compilation..." -ForegroundColor Yellow
    
    try {
        # Create output directory
        if (!(Test-Path "output")) {
            New-Item -ItemType Directory -Name "output" | Out-Null
        }
        
        # Compilation passes
        Write-Host "Pass 1/3: Initial compilation..." -ForegroundColor Yellow
        $result1 = Start-Process -FilePath "pdflatex" -ArgumentList "-output-directory=output", "-interaction=nonstopmode", "main.tex" -Wait -PassThru -NoNewWindow
        
        Write-Host "Pass 2/3: Processing cross-references..." -ForegroundColor Yellow
        $result2 = Start-Process -FilePath "pdflatex" -ArgumentList "-output-directory=output", "-interaction=nonstopmode", "main.tex" -Wait -PassThru -NoNewWindow
        
        Write-Host "Pass 3/3: Final compilation..." -ForegroundColor Yellow
        $result3 = Start-Process -FilePath "pdflatex" -ArgumentList "-output-directory=output", "-interaction=nonstopmode", "main.tex" -Wait -PassThru -NoNewWindow
        
        # Check if PDF was created
        if (Test-Path "output\main.pdf") {
            $fileSize = (Get-Item "output\main.pdf").Length / 1MB
            Write-Host "✓ Compilation successful!" -ForegroundColor Green
            Write-Host "✓ Output: output\main.pdf ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
            
            # Open PDF
            $open = Read-Host "Open the compiled PDF? (y/n)"
            if ($open -eq "y" -or $open -eq "Y") {
                Start-Process "output\main.pdf"
            }
            
            return $true
        } else {
            Write-Host "✗ PDF generation failed" -ForegroundColor Red
            Write-Host "Check output\main.log for error details" -ForegroundColor Yellow
            return $false
        }
        
    } catch {
        Write-Host "✗ Compilation failed: $_" -ForegroundColor Red
        return $false
    }
}

# Function to clean auxiliary files
function Remove-AuxiliaryFiles {
    Write-Host "Cleaning auxiliary files..." -ForegroundColor Yellow
    
    $extensions = @("*.aux", "*.log", "*.toc", "*.lof", "*.lot", "*.out", "*.fls", "*.fdb_latexmk", "*.bbl", "*.blg")
    foreach ($ext in $extensions) {
        Get-ChildItem -Path . -Name $ext | Remove-Item -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path "output" -Name $ext -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "✓ Cleanup completed" -ForegroundColor Green
}

# Main execution
try {
    if ($CleanOnly) {
        Remove-AuxiliaryFiles
        exit 0
    }
    
    # Check current directory
    if (!(Test-Path "main.tex")) {
        Write-Host "✗ main.tex not found. Please run this script from the latex directory." -ForegroundColor Red
        exit 1
    }
    
    # Check/Install LaTeX
    if (!(Test-LaTeXInstallation) -and !$SkipInstall) {
        Write-Host "LaTeX not found. Attempting to install TinyTeX..." -ForegroundColor Yellow
        if (!(Install-TinyTeX)) {
            Write-Host "Failed to install LaTeX. Please install manually:" -ForegroundColor Red
            Write-Host "1. Visit: https://miktex.org/download" -ForegroundColor Yellow
            Write-Host "2. Download and install MiKTeX" -ForegroundColor Yellow
            Write-Host "3. Run this script again with -SkipInstall flag" -ForegroundColor Yellow
            exit 1
        }
    }
    
    # Verify LaTeX is available
    if (!(Test-LaTeXInstallation)) {
        Write-Host "✗ LaTeX still not available. Please install manually." -ForegroundColor Red
        exit 1
    }
    
    # Compile document
    if (Invoke-LaTeXCompilation) {
        Write-Host "==========================================" -ForegroundColor Cyan
        Write-Host "CloudForge AI Technical Report - READY!" -ForegroundColor Cyan
        Write-Host "==========================================" -ForegroundColor Cyan
    } else {
        exit 1
    }
    
} catch {
    Write-Host "✗ Script execution failed: $_" -ForegroundColor Red
    exit 1
} finally {
    # Always clean up auxiliary files
    Remove-AuxiliaryFiles
}