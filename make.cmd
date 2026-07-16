@echo off
rem Thin wrapper so `.\make.cmd <target> [KEY=value ...]` works from cmd.exe
rem or PowerShell without ever hitting PowerShell's script execution policy:
rem .cmd/.bat files aren't subject to it, only .ps1 files are. See make.ps1
rem for the actual implementation.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make.ps1" %*
