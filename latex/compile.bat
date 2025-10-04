@echo off
echo CloudForge AI - LaTeX Compilation Script
echo =======================================

echo.
echo Checking LaTeX installation...
pdflatex --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pdflatex not found!
    echo Please install MiKTeX from: https://miktex.org/download
    echo.
    echo Alternative: Use Overleaf online compiler
    echo 1. Go to https://www.overleaf.com
    echo 2. Create new project
    echo 3. Upload all files maintaining folder structure
    pause
    exit /b 1
)

echo LaTeX found! Starting compilation...
echo.

echo Pass 1: Initial compilation...
pdflatex -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo ERROR: First compilation failed!
    echo Check the log file for errors.
    pause
    exit /b 1
)

echo Pass 2: Processing references...
pdflatex -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo ERROR: Second compilation failed!
    echo Check the log file for errors.
    pause
    exit /b 1
)

echo Pass 3: Final compilation...
pdflatex -interaction=nonstopmode main.tex
if %errorlevel% neq 0 (
    echo ERROR: Final compilation failed!
    echo Check the log file for errors.
    pause
    exit /b 1
)

echo.
echo =======================================
echo SUCCESS: PDF generated successfully!
echo Output file: main.pdf
echo =======================================
echo.

if exist main.pdf (
    echo Opening PDF...
    start main.pdf
) else (
    echo Warning: PDF file not found despite successful compilation.
)

echo.
echo Cleaning up auxiliary files...
del *.aux *.log *.toc *.lof *.lot *.out *.fls *.fdb_latexmk 2>nul

echo Compilation complete!
pause