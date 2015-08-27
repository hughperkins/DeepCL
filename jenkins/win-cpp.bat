set WINBITS=%1
echo WINBITS: %WINBITS%

cd %~dp0.
for /f "" %%i in (version.txt) do (
   set version=%%i
)
cd ..

if not exist turbogjpeg-win%WINBITS%.zip powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win%WINBITS%-static.zip', 'turbojpeg-win%WINBITS%.zip')
if errorlevel 1 exit /B 1
rmdir /s /q turbojpeg-win%WINBITS%
mkdir turbojpeg-win%WINBITS%
cd turbojpeg-win%WINBITS%
"c:\program files\7-Zip\7z.exe" x ..\turbojpeg-win%WINBITS%.zip
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
echo get_filename_component(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)>initcache.cmake
echo set(JPEG_INCLUDE_DIR "${SOURCE_DIR}/turbojpeg-win%WINBITS%" CACHE PATH "JPEG_INCLUDE_DIR")>>initcache.cmake
echo set(JPEG_LIBRARY "${SOURCE_DIR}/turbojpeg-win%WINBITS%/turbojpeg-static.lib" CACHE PATH "JPEG_LIBRARY")>>initcache.cmake
if exist "c:\program files\cmake\bin\cmake.exe" set "CMAKEEXE=c:\program files\cmake\bin\cmake.exe"
if exist "c:\program files (x86)\cmake\bin\cmake.exe" set "CMAKEEXE=c:\program files (x86)\cmake\bin\cmake.exe"
set "generatorpostfix="
if x%WINBITS%==x64 set "generatorpostfix= Win64"
"%CMAKEEXE%" -G "Visual Studio 10 2010%generatorpostfix%" -C initcache.cmake ..
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

copy vc2010redist\win%WINBITS%\* Release

cd ..
"c:\program files\7-Zip\7z.exe" a deepcl-win%WINBITS%-%version%.zip dist
if errorlevel 1 exit /B 1

cd ..
echo %version%>latestUnstable.txt

