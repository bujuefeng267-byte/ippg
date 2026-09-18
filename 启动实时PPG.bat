@echo off
chcp 65001 >nul
echo Open http://127.0.0.1:8765 in Chrome or Edge after the service starts.
echo If the service is already running, open that address directly.
wsl -d Ubuntu -u fengbujue --exec bash /home/fengbujue/项目/rppg识别/run_realtime.sh
pause
