call %~dp0win-py.bat 32 35
if errorlevel 1 goto :fail
goto :eof

:fail
echo Something went wrong
exit /B 1

