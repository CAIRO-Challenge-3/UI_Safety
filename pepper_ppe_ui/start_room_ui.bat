@echo off
rem Starts the Pepper room UI in your browser.
rem   Double-click            -> Werkstattbereich 1
rem   start_room_ui.bat speisesaal   (werkstatt-1 | werkstatt-2 | speisesaal | sanitaer)
rem Close the "Pepper UI server" window to stop.

set ROOM=%1
if "%ROOM%"=="" set ROOM=werkstatt-1
set PORT=8000

cd /d "%~dp0html"
start "Pepper UI server" python -m http.server %PORT%
timeout /t 2 /nobreak >nul
start "" "http://localhost:%PORT%/room.html?room=%ROOM%"
