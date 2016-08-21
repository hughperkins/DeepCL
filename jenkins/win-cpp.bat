set WINBITS=%1
echo WINBITS: %WINBITS%

cd %~dp0.
for /f "" %%i in (version.txt) do (
   set version=%%i
)

cd %~dp0..
if not exist turbogjpeg-win%WINBITS%.zip powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://deepcl.hughperkins.com/Downloads/jpeg-9b-bin.zip', 'jpeg-9b-bin.zip')
if errorlevel 1 exit /B 1
rmdir /s /q jpeg-9b-bin
"c:\program files\7-Zip\7z.exe" x jpeg-9b-bin.zip
if errorlevel 1 exit /B 1

cd %~dp0..
rem dir
rmdir /s /q build
mkdir build
cd %~dp0..\build
rem dir
set "VS100COMNTOOLS=c:\Program Files (x86)\Microsoft Visual Studio 10.0\Common7\Tools\"
set "VS110COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio 11.0\Common7\Tools\"
set "VS120COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio 12.0\Common7\Tools\"
echo get_filename_component(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)>initcache.cmake
echo set(JPEG_INCLUDE_DIR "${SOURCE_DIR}/jpeg-9b-bin" CACHE PATH "JPEG_INCLUDE_DIR")>>initcache.cmake
echo set(JPEG_LIBRARY "${SOURCE_DIR}/jpeg-9b-bin/win%WINBITS%/jpeg.lib" CACHE PATH "JPEG_LIBRARY")>>initcache.cmake
if exist "c:\program files\cmake\bin\cmake.exe" set "CMAKEEXE=c:\program files\cmake\bin\cmake.exe"
if exist "c:\program files (x86)\cmake\bin\cmake.exe" set "CMAKEEXE=c:\program files (x86)\cmake\bin\cmake.exe"
set "generatorpostfix="
if x%WINBITS%==x64 set "generatorpostfix= Win64"
"%CMAKEEXE%" -G "Visual Studio 14 2015%generatorpostfix%" -C initcache.cmake ..
C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe ALL_BUILD.vcxproj /p:Configuration=Release
if errorlevel 1 exit /B 1
C:\WINDOWS\Microsoft.NET\Framework\v4.0.30319\MSBuild.exe INSTALL.vcxproj /p:Configuration=Release
if errorlevel 1 exit /B 1

rem copy down the redistributables (maybe they're on the server somewhere?)
cd %~dp0..
powershell Set-ExecutionPolicy unrestricted
if errorlevel 1 exit /B 1

rem if not exist vc2010redist.zip powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://deepcl.hughperkins.com/Downloads/vc2010redist.zip', 'vc2010redist.zip')
rem if errorlevel 1 exit /B 1

rem rmdir /s /q vc2010redist
rem "c:\program files\7-Zip\7z.exe" x vc2010redist.zip
rem if errorlevel 1 exit /B 1

rem copy vc2010redist\win%WINBITS%\* dist\bin
rem copy "jpegturbo-1.5-%WINBITS%\jpeg62.dll" dist\bin

if not exist msvc2015-win%WINBITS%.zip powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://deepcl.hughperkins.com/Downloads/msvc2015-win%WINBITS%.zip', 'msvc2015-win%WINBITS%.zip')
if errorlevel 1 exit /B 1
rmdir /s /q msvc2015-win%WINBITS%
"c:\program files\7-Zip\7z.exe" x msvc2015-win%WINBITS%.zip
if errorlevel 1 exit /B 1
copy msvc2015-win%WINBITS%\* dist\bin

cd %~dp0..
"c:\program files\7-Zip\7z.exe" a deepcl-win%WINBITS%-%version%.zip dist
if errorlevel 1 exit /B 1

cd %~dp0..
echo %version%>latestUnstable.txt
