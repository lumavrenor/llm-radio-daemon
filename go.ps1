# Start LLM Radio Daemon
#
# The <lang> argument is required and names the config\<lang>\ directory to
# load. There is deliberately no default, so the command line always shows
# which language is on air.
#
#   go.bat ja
#   go.bat en
#   go.bat en --log-level DEBUG   remaining args pass through to main.py
#
# If .venv doesn't exist yet, this script creates it, installs the
# dependencies from requirements.txt, and then launches.
#
# If config\<lang>\config.toml / config_cast.toml / config_content.toml don't
# exist yet, they are auto-copied from the matching .example file (existing
# files are never overwritten).
#
# This is the actual implementation invoked by go.bat. go.bat itself stays
# an ASCII-only thin shim, because cmd.exe's batch parser is fragile with
# non-ASCII text -- the real logic lives here instead.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$lang = $args | Select-Object -First 1
if (-not $lang -or $lang.StartsWith('-')) {
    Write-Host "usage: go.bat <lang> [args...]"
    Write-Host ""
    Write-Host "  <lang> is a directory name under config\. Currently available:"
    Get-ChildItem -Path (Join-Path $PSScriptRoot 'config') -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "    $($_.Name)" }
    Write-Host ""
    Write-Host "  example: go.bat ja"
    Write-Host "           go.bat en --log-level DEBUG"
    exit 2
}

$rest = @()
if ($args.Count -gt 1) {
    $rest = $args[1..($args.Count - 1)]
}

$configDir = Join-Path $PSScriptRoot "config\$lang"
foreach ($f in @('config.toml', 'config_cast.toml', 'config_content.toml')) {
    $dst = Join-Path $configDir $f
    $src = Join-Path $configDir "$f.example"
    if ((-not (Test-Path $dst)) -and (Test-Path $src)) {
        Copy-Item $src $dst
        Write-Host "[first-time setup] Created $f from $f.example."
    }
}

$pyExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pyExe)) {
    Write-Host "[first-time setup]"
    Write-Host "Virtual environment (.venv) not found. Creating it..."

    python -m venv (Join-Path $PSScriptRoot '.venv')
    if (-not (Test-Path $pyExe)) {
        Write-Host "[error]"
        Write-Host "Failed to create .venv. Make sure Python is installed and on PATH."
        exit 1
    }

    Write-Host "Installing dependencies (pip install -r requirements.txt)..."
    & $pyExe -m pip install -q --upgrade pip
    & $pyExe -m pip install -q -r (Join-Path $PSScriptRoot 'requirements.txt')
    Write-Host "Done."
}

# Language-specific extra packages (e.g. requirements-en.txt). Installed here
# regardless of which language .venv was first created for, so switching
# languages later still picks up what that language needs.
$extraReq = Join-Path $PSScriptRoot "requirements-$lang.txt"
$extraMarker = Join-Path $PSScriptRoot ".venv\.extras-$lang-installed"
if ((Test-Path $extraReq) -and (-not (Test-Path $extraMarker))) {
    Write-Host "Installing $lang-specific dependencies (pip install -r requirements-$lang.txt)..."
    & $pyExe -m pip install -q -r $extraReq
    if ($lang -eq 'en') {
        Write-Host "Downloading spaCy model (en_core_web_sm)..."
        & $pyExe -m spacy download en_core_web_sm
    }
    New-Item -ItemType File -Path $extraMarker -Force | Out-Null
    Write-Host "Done."
}

& $pyExe -m llm_radio_daemon.main --config "config\$lang\config.toml" @rest
exit $LASTEXITCODE
