# CloudForge AI LaTeX Compilation Script
# This script compiles the complete technical report

Write-Host "CloudForge AI - Comprehensive Technical Report Compilation" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Check if pdflatex is available
if (!(Get-Command "pdflatex" -ErrorAction SilentlyContinue)) {
    Write-Host "Error: pdflatex not found. Please install a LaTeX distribution." -ForegroundColor Red
    exit 1
}

# Create output directory
if (!(Test-Path "output")) {
    New-Item -ItemType Directory -Name "output"
    Write-Host "Created output directory" -ForegroundColor Green
}

Write-Host "Starting compilation process..." -ForegroundColor Yellow

try {
    # First pass - generate aux files
    Write-Host "Pass 1: Generating auxiliary files..." -ForegroundColor Yellow
    pdflatex -output-directory=output main.tex > $null
    
    # Generate bibliography
    Write-Host "Pass 2: Processing bibliography..." -ForegroundColor Yellow
    Set-Location output
    bibtex main > $null
    Set-Location ..
    
    # Second pass - resolve references
    Write-Host "Pass 3: Resolving cross-references..." -ForegroundColor Yellow
    pdflatex -output-directory=output main.tex > $null
    
    # Third pass - finalize document
    Write-Host "Pass 4: Finalizing document..." -ForegroundColor Yellow
    pdflatex -output-directory=output main.tex > $null
    
    Write-Host "Compilation completed successfully!" -ForegroundColor Green
    Write-Host "Output: output/main.pdf" -ForegroundColor Green
    
    # Check if PDF was created
    if (Test-Path "output/main.pdf") {
        $fileSize = (Get-Item "output/main.pdf").Length / 1MB
        Write-Host "Document size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green
        
        # Optionally open the PDF
        $openPdf = Read-Host "Open the compiled PDF? (y/n)"
        if ($openPdf -eq "y" -or $openPdf -eq "Y") {
            Start-Process "output/main.pdf"
        }
    }
    
} catch {
    Write-Host "Compilation failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "CloudForge AI Technical Report - Ready for Distribution" -ForegroundColor Cyan