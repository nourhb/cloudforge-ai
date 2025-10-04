# 📚 CloudForge AI Documentation Sync

This directory contains automation scripts to keep LaTeX documentation synchronized with application changes.

## 🚀 Quick Usage

### Manual Sync (Recommended)
```powershell
# Sync documentation and push to GitHub
.\scripts\sync-docs.ps1

# Sync with custom commit message
.\scripts\sync-docs.ps1 -Message "feat: add new billing module with documentation"

# Force sync (resolves conflicts automatically)
.\scripts\sync-docs.ps1 -Force
```

### Automatic Watching
```powershell
# Start file watcher (runs in background)
.\scripts\watch-changes.ps1
```

### NPM Scripts
```bash
npm run sync-docs          # Manual sync
npm run sync-docs-force    # Force sync
npm run watch-docs         # Start file watcher
npm run compile-latex      # Just compile PDF
```

## 🔄 How It Works

### 1. **Automatic Detection**
- Monitors changes in `backend/`, `frontend/`, `ai-scripts/`, `helm-chart/`, `infra/`
- Detects which components were modified
- Updates corresponding LaTeX chapters with timestamps

### 2. **Smart Documentation Updates**
- **Backend changes** → Updates `chapters/04_architecture.tex`
- **Frontend changes** → Updates `chapters/07_sprint_03.tex`  
- **AI/ML changes** → Updates `chapters/06_sprint_02.tex`
- **Infrastructure changes** → Updates `chapters/17_deployment.tex`

### 3. **LaTeX Compilation**
- Automatically compiles `main_fixed.tex` to PDF
- Validates successful compilation (78+ pages expected)
- Handles TikZ diagrams and charts

### 4. **GitHub Integration**
- Creates intelligent commit messages based on changes
- Pushes both app changes AND updated documentation
- Includes PDF in the repository

## 🤖 GitHub Actions

The workflow automatically runs on every push to `main` branch:

- **Triggers:** Changes to backend, frontend, ai-scripts, helm-chart, infra
- **Actions:** Updates docs, compiles PDF, commits, and pushes
- **Reports:** Generates summary of what was updated

## 📁 File Structure

```
scripts/
├── sync-docs.ps1           # Main sync script
├── watch-changes.ps1       # File watcher
└── README.md              # This file

.github/workflows/
└── sync-docs.yml          # GitHub Actions workflow

latex/
├── main_fixed.tex         # Main LaTeX document
├── chapters/              # Individual chapters
└── main_fixed.pdf         # Compiled output (78+ pages)
```

## ⚙️ Configuration

### Pre-commit Hook (Optional)
The pre-commit hook automatically compiles LaTeX when `.tex` files are modified:

```bash
# Make executable (if not already)
chmod +x .git/hooks/pre-commit
```

### Environment Requirements
- **PowerShell 5.1+** (Windows)
- **Git** (configured with GitHub access)
- **MiKTeX/LaTeX** (for PDF compilation)
- **Node.js** (for npm scripts, optional)

## 🎯 Best Practices

1. **Always run sync before major commits:**
   ```powershell
   .\scripts\sync-docs.ps1 -Message "feat: implement user authentication system"
   ```

2. **Use the file watcher during development:**
   ```powershell
   .\scripts\watch-changes.ps1
   # Leave running in background
   ```

3. **Check PDF compilation:**
   - Expected: 78+ pages
   - Size: ~650KB
   - Should include diagrams and charts

4. **Commit message format:**
   - `feat:` for new features
   - `fix:` for bug fixes  
   - `docs:` for documentation-only changes
   - `chore:` for maintenance

## 🔧 Troubleshooting

### LaTeX Compilation Fails
```powershell
cd latex
pdflatex -interaction=nonstopmode main_fixed.tex
# Check the .log file for specific errors
```

### Git Push Fails
```powershell
# Force sync with conflict resolution
.\scripts\sync-docs.ps1 -Force
```

### File Watcher Not Working
```powershell
# Check PowerShell execution policy
Get-ExecutionPolicy
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📊 Status Indicators

- ✅ **Green:** Successful sync and push
- ⚠️ **Yellow:** Warnings but completed
- ❌ **Red:** Failed - requires attention
- 🔄 **Blue:** Processing/compiling
- 📚 **Purple:** Documentation updated