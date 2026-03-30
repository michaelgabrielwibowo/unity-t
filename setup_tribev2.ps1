# ===================================================================
# TRIBE v2 Streaming Stack - Setup Script (PowerShell)
# ===================================================================

$ErrorActionPreference = "Continue"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  TRIBE v2 Streaming Stack - Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/6] Checking Python..." -ForegroundColor Yellow
$pythonVersion = & python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

# 2. Create virtual environment
$venvPath = Join-Path $PSScriptRoot "venv"
Write-Host "`n[2/6] Creating virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path $venvPath)) {
    & python -m venv $venvPath
    Write-Host "  Created: $venvPath" -ForegroundColor Green
} else {
    Write-Host "  Already exists: $venvPath" -ForegroundColor Green
}

# Activate
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
. $activateScript

# 3. Install TRIBE v2
Write-Host "`n[3/6] Installing TRIBE v2 and dependencies..." -ForegroundColor Yellow

$tribev2Path = Join-Path $PSScriptRoot "tribev2"
if (Test-Path $tribev2Path) {
    & pip install -e "$tribev2Path" --quiet
    Write-Host "  TRIBE v2 installed" -ForegroundColor Green
} else {
    Write-Host "  WARNING: tribev2/ not found." -ForegroundColor Red
}

# 4. Install streaming dependencies
Write-Host "`n[4/6] Installing streaming and OSC dependencies..." -ForegroundColor Yellow

& pip install python-osc opencv-python sounddevice soundfile pyyaml --quiet
Write-Host "  Core streaming deps installed" -ForegroundColor Green

# Optional: Whisper for ASR
Write-Host "  Installing Whisper ASR (optional)..." -ForegroundColor Gray
& pip install openai-whisper --quiet 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Whisper installed" -ForegroundColor Green
} else {
    Write-Host "  Whisper install skipped (optional)" -ForegroundColor Gray
}

# 5. HuggingFace Authentication
Write-Host "`n[5/6] HuggingFace authentication..." -ForegroundColor Yellow
Write-Host "  TRIBE v2 uses LLaMA 3.2-3B (gated model)" -ForegroundColor Gray
Write-Host "  Run huggingface-cli login if needed" -ForegroundColor Gray

$hfResult = & huggingface-cli whoami 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Logged in as: $hfResult" -ForegroundColor Green
} else {
    Write-Host "  Not logged in yet" -ForegroundColor Yellow
}

# 6. Verify installation
Write-Host "`n[6/6] Verifying installation..." -ForegroundColor Yellow
$verifyPath = Join-Path $PSScriptRoot "tools\verify_install.py"
& python $verifyPath

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Activate venv: .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  2. Login to HF: huggingface-cli login" -ForegroundColor Gray
Write-Host "  3. Export mesh: python tools\export_fsaverage5_mesh.py" -ForegroundColor Gray
Write-Host "  4. Run streaming: python run_tribe_stream.py" -ForegroundColor Gray
Write-Host ""
