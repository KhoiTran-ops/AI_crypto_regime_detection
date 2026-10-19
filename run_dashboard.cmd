@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv. Follow the installation steps in README.md.
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run dashboard.py --server.address 127.0.0.1 %*
