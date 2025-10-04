@echo off
echo CloudForge AI - Local LaTeX Compilation Setup
echo ==============================================
echo.

echo Step 1: Download and Install MiKTeX
echo -----------------------------------
echo Please follow these steps:
echo 1. Go to: https://miktex.org/download
echo 2. Download "Basic MiKTeX Installer" for Windows
echo 3. Run the installer with default settings
echo 4. Restart this command prompt after installation
echo.

echo Step 2: After MiKTeX Installation
echo ---------------------------------
echo Run this batch file again to compile your document
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>&1
if %errorlevel% equ 0 (
    echo MiKTeX detected! Starting compilation...
    echo.
    
    echo Compiling CloudForge AI Technical Report...
    echo Pass 1/3: Initial compilation
    pdflatex -interaction=nonstopmode main.tex
    
    echo Pass 2/3: Processing references
    pdflatex -interaction=nonstopmode main.tex
    
    echo Pass 3/3: Final compilation
    pdflatex -interaction=nonstopmode main.tex
    
    if exist main.pdf (
        echo.
        echo =======================================
        echo SUCCESS: PDF generated successfully!
        echo File: main.pdf
        echo =======================================
        echo Opening PDF...
        start main.pdf
        
        echo Cleaning up auxiliary files...
        del *.aux *.log *.toc *.lof *.lot *.out *.fls *.fdb_latexmk 2>nul
    ) else (
        echo ERROR: PDF compilation failed!
        echo Check main.log for error details.
    )
) else (
    echo MiKTeX not found. Please install MiKTeX first.
    echo Opening download page...
    start https://miktex.org/download
)

echo.
pause