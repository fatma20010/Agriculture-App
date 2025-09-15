@echo off
echo Starting Inveepplant servers...
echo.

echo Starting Farm server with forwarding capabilities (veg.py)...
start cmd /k "python -m uvicorn veg:app --host 0.0.0.0 --port 5002 --reload"
timeout /t 5

echo Starting the grass server (grass.py)...
start cmd /k "python -m uvicorn grass:app --host 0.0.0.0 --port 5000 --reload"
timeout /t 5

echo Starting frontend...
start cmd /k "npm run dev"

echo.
echo All servers have been started.
echo You can access the app at http://localhost:8097
echo.
echo Press any key to exit this window (servers will continue running)
pause > nul 