@echo off
rem mytetration real-time preview viewer: double-click to open (drag = move, wheel = zoom, close the window to quit).
rem Opens at once; it only refuses when the chosen GPU has less than 2,048 MiB free (the reason is printed).
rem GPU: nvidia-smi index in the environment variable MYTET_GPU (default 0), e.g. "set MYTET_GPU=1" before running.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m viewer --interactive
if errorlevel 1 pause
