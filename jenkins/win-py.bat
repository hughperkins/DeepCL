echo args: %1

call \%1\scripts\activate

python -c "from __future__ import print_function; import platform; print( platform.uname() )"
python -c "from __future__ import print_function; import platform; print( platform.architecture() )"

cd
rmdir /s /q build
rmdir /s /q dist
mkdir build
cd build
"c:\program files (x86)\cmake\bin\cmake" -G "Visual Studio 10 2010" ..
C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe ALL_BUILD.vcxproj /p:Configuration=Release
C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe INSTALL.vcxproj /p:Configuration=Release
if errorlevel 1 exit /B 1
cd ..
cd
dir
call dist\bin\activate.bat

copy /y jenkins\version.txt python
cd python

rmdir /s /q dist
rmdir /s /q build
rmdir /s /q mysrc
rmdir /s /q src
dir
if exist dist goto :error
if exist build goto :error
if exist mysrc goto :error
if exist src goto :error

python setup.py build_ext -i
if errorlevel 1 goto :error

python setup.py bdist_egg
if errorlevel 1 goto :error

set HOME=%HOMEPATH%
python setup.py bdist_egg upload
rem ignore any error?
rem if errorlevel 1 goto :error

exit /B 0

goto :eof

:error
echo Error occurred
exit /B 1

