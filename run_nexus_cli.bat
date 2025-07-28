@echo off
title Nexus CLI - Intelligent Coding Assistant
echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                      NEXUS CLI                               ║
echo ║              Intelligent Coding Assistant                    ║
echo ║              Powered by iLLuMinator-4.7B                     ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

if exist "dist\nexus-cli\nexus-cli.exe" (
    echo Starting Nexus CLI...
    "dist\nexus-cli\nexus-cli.exe"
) else if exist "dist\nexus-cli-standalone.exe" (
    echo Starting Nexus CLI (standalone)...
    "dist\nexus-cli-standalone.exe"
) else (
    echo ERROR: Nexus CLI executable not found!
    echo Please build the executable first by running: python build_executable.py
    pause
)
