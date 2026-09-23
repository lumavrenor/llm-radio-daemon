# Send a request to the running broadcast
#
# The language argument is REQUIRED: it names the config\<lang>\ directory
# to load. There is deliberately no default -- picking the wrong language
# writes to the wrong database (ja: data\llm_radio_daemon.ja.db / en: data\llm_radio_daemon.en.db).
#
#   request.bat ja
#   request.bat en
#
# Common uses (see llm_radio_daemon/request.py for the full list):
#
#   request.bat ja now --list           list the corners you can request
#   request.bat ja now arxiv            switch to that corner now (30 min by default)
#   request.bat ja now arxiv --for 2h   ... for a different length
#   request.bat ja now --clear          drop the request, back to the timetable
#   request.bat ja reset all            wipe read/broadcast state, back to "nothing seen yet"
#   request.bat ja reset data           relocate & recreate all of data\ (last resort; stop the broadcast first)
#
# This is the actual implementation invoked by request.bat. See the comment
# at the top of go.ps1 for why the logic lives here instead of in the .bat.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# Keep the window open only when launched from Explorer (double-click), where
# it would otherwise close before the output can be read. From a terminal the
# prompt is just noise, so skip it there.
function Wait-IfLaunchedFromExplorer {
    try {
        $id = $PID
        for ($i = 0; $i -lt 2; $i++) {
            $id = (Get-CimInstance Win32_Process -Filter "ProcessId=$id").ParentProcessId
            $name = (Get-CimInstance Win32_Process -Filter "ProcessId=$id").Name
            if ($name -eq 'explorer.exe') {
                Read-Host "Press Enter to exit" | Out-Null
                return
            }
        }
    } catch {}
}

$lang = $args | Select-Object -First 1
if (-not $lang -or $lang.StartsWith('-')) {
    Write-Host "usage: request.bat <lang> [args...]"
    Write-Host ""
    Write-Host "  <lang> is a directory name under config\. Currently available:"
    Get-ChildItem -Path (Join-Path $PSScriptRoot 'config') -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "    $($_.Name)" }
    Write-Host ""
    Write-Host "  example: request.bat ja now --list"
    Write-Host "           request.bat ja now arxiv --for 30m"
    Wait-IfLaunchedFromExplorer
    exit 2
}

$rest = @()
if ($args.Count -gt 1) {
    $rest = $args[1..($args.Count - 1)]
}

$pyExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pyExe)) {
    Write-Host "[error]"
    Write-Host "Virtual environment (.venv) not found. Run `"go.bat $lang`" once first."
    Wait-IfLaunchedFromExplorer
    exit 1
}

& $pyExe -m llm_radio_daemon.request --config "config\$lang\config.toml" @rest
$code = $LASTEXITCODE
Wait-IfLaunchedFromExplorer
exit $code
