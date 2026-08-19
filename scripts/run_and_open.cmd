@echo off
setlocal

set "ROOT=%~dp0.."
cd /d "%ROOT%"

".venv\Scripts\python.exe" -m pmreport --config config.yaml --verbose
if errorlevel 1 (
    echo.
    echo Report generation failed. Check logs\daily.log.
    pause
    exit /b 1
)

for /f "usebackq tokens=*" %%D in (`powershell -NoProfile -Command "(Get-Date).ToString('yyyy-MM-dd')"`) do set "REPORT_DATE=%%D"
set "REPORT_HTML=%ROOT%\reports\%REPORT_DATE%\metals-report-%REPORT_DATE%.html"

if exist "%REPORT_HTML%" (
    start "" "%REPORT_HTML%"
) else (
    echo Generated report file not found:
    echo %REPORT_HTML%
    pause
)

endlocal
