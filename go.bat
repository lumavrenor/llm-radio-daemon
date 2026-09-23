@echo off
rem Start LLM Radio Daemon. This is a thin ASCII-only shim: cmd.exe's batch
rem parser is fragile with non-ASCII (Japanese) text, so the real logic and
rem bilingual (JA/EN) messages live in go.ps1, run here via PowerShell.
rem See go.ps1 for the actual usage docs and behavior.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0go.ps1" %*
exit /b %ERRORLEVEL%
