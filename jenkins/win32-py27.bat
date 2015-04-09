call %~dp0win-py.bat env-27-32
if errorlevel 1 goto :fail
goto :eof

:fail
echo Something went wrong
exit /B 1

