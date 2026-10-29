@echo off
rem GSplat Studio - double-click once on a new machine.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
pause
