@echo off
cd /d "%~dp0"
echo ========================================================
echo  Starting QEC Surface Code and RL Decoder Dashboard...
echo ========================================================
python -m streamlit run scripts/dashboard.py
pause

