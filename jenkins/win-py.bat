echo args: %1 %2
set bitness=%1
set pyversion=%2
echo bitness: %bitness%
echo pyversion: %pyversion%

call %~dp0win-cpp.bat %bitness%

call \py%pyversion%-%bitness%\scripts\activate

python -c "from __future__ import print_function; import platform; print( platform.uname() )"
python -c "from __future__ import print_function; import platform; print( platform.architecture() )"

rem  cd
rem  rmdir /s /q build
rem  rmdir /s /q dist
rem  dir
rem  mkdir build
rem  cd build
rem  set "generatorpostfix="
rem  if %bitness%==64 set "generatorpostfix= Win64"
rem  "c:\program files (x86)\cmake\bin\cmake" -G "Visual Studio 10 2010%generatorpostfix%" ..
rem  C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe ALL_BUILD.vcxproj /p:Configuration=Release
rem  if errorlevel 1 exit /B 1
rem  C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe INSTALL.vcxproj /p:Configuration=Release
rem  if errorlevel 1 exit /B 1
rem  cd ..
rem  cd
rem  dir
cd %~dp0..
call dist\bin\activate.bat
pip install numpy
pip install pytest

copy /y jenkins\version.txt python
cd python

rmdir /s /q dist
rmdir /s /q build
rmdir /s /q mysrc
rmdir /s /q src

if exist dist goto :error
if exist build goto :error
if exist mysrc goto :error
if exist src goto :error

python setup.py build_ext -i
if errorlevel 1 goto :error

python setup.py bdist_egg
if errorlevel 1 goto :error

python setup.py install
if errorlevel 1 goto :error
py.test -sv test
if errorlevel 1 goto :error
set HOME=c:\Users\Administrator
echo HOME: %HOME%
python setup.py bdist_egg upload
rem ignore any error?
rem if errorlevel 1 goto :error

exit /B 0

goto :eof

:error
echo Error occurred
exit /B 1
