cd jenkins
for /f "" %%i in (version.txt) do (
   set version=%%i
)
cd ..
dir
rmdir /s /q build
mkdir -p build
cd build
dir
"c:\program files (x86)\cmake\bin\cmake" -G "Visual Studio 10 2010 Win64" ..
C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe ALL_BUILD.vcxproj /p:Configuration=Release
if errorlevel 1 exit /B 1
cd Release
"c:\program files\7-Zip\7z.exe" a deepcl-win64-%version%.zip *
if errorlevel 1 exit /B 1

