@echo off
rem Send a request to the running broadcast. This is a thin ASCII-only shim:
rem cmd.exe's batch parser is fragile with non-ASCII (Japanese) text, so the
rem real logic and bilingual (JA/EN) messages live in request.ps1, run here
rem via PowerShell. See request.ps1 for the actual usage docs and behavior.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0request.ps1" %*
exit /b %ERRORLEVEL%
