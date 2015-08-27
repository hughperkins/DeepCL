cd jenkins
for /f "" %%i in (version.txt) do (
   set version=%%i
)
cd ..

powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win64-static.zip', 'turbojpeg-win64.zip')
if errorlevel 1 exit /B 1
mkdir turbojpeg-win64
cd turbojpeg-win64
"c:\program files\7-Zip\7z.exe" x ..\turbojpeg-win64.zip
if errorlevel 1 exit /B 1
cd ..

dir
rmdir /s /q build
mkdir build
cd build
dir
set "VS100COMNTOOLS=c:\Program Files (x86)\Microsoft Visual Studio 10.0\Common7\Tools\"
set "VS110COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio 11.0\Common7\Tools\"
set "VS120COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio 12.0\Common7\Tools\"
"c:\program files (x86)\cmake\bin\cmake" -G "Visual Studio 10 2010 Win64" -D BUILD_PYSWIG_WRAPPERS:BOOL=OFF -D BUILD_LUA_WRAPPERS:BOOL=OFF -D JPEG_INCLUDE_DIR=%CD%\..\turbojpeg-win64 -D JPEG_LIBRARY=%CD%\..\turbojpeg-win64\turbojpeg-static.lib ..
C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe ALL_BUILD.vcxproj /p:Configuration=Release
if errorlevel 1 exit /B 1
C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe INSTALL.vcxproj /p:Configuration=Release
if errorlevel 1 exit /B 1

rem copy down the redistributables (maybe they're on the server somewhere?)
powershell Set-ExecutionPolicy unrestricted
powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://deepcl.hughperkins.com/Downloads/vc2010redist.zip', 'vc2010redist.zip')
if errorlevel 1 exit /B 1

"c:\program files\7-Zip\7z.exe" x vc2010redist.zip
if errorlevel 1 exit /B 1

copy vc2010redist\win64\* Release

cd ..
"c:\program files\7-Zip\7z.exe" a deepcl-win64-%version%.zip dist
if errorlevel 1 exit /B 1

cd ..
echo %version%>latestUnstable.txt

