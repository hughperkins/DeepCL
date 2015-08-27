cd jenkins
for /f "" %%i in (version.txt) do (
   set version=%%i
)
cd ..

powershell.exe -Command (new-object System.Net.WebClient).DownloadFile('http://deepcl.hughperkins.com/Downloads/turbojpeg-1.4.0-win32-static.zip', 'turbojpeg-win32.zip')
if errorlevel 1 exit /B 1
rmdir /s /q turbojpeg-win32
mkdir turbojpeg-win32
cd turbojpeg-win32
"c:\program files\7-Zip\7z.exe" x ..\turbojpeg-win32.zip
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
echo set(BUILD_LUA_WRAPPERS "OFF" CACHE BOOL "BUILD_LUA_WRAPPERS")>initcache.cmake
echo get_filename_component(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)>>initcache.cmake
echo set(JPEG_INCLUDE_DIR "${SOURCE_DIR}/turbojpeg-win32" CACHE PATH "JPEG_INCLUDE_DIR")>>initcache.cmake
echo set(JPEG_LIBRARY "${SOURCE_DIR}/turbojpeg-win32/turbojpeg-static.lib" CACHE PATH "JPEG_LIBRARY")>>initcache.cmake
if exist "c:\program files\cmake\bin\cmake.exe" set CMAKEEXE="c:\program files\cmake\bin\cmake.exe"
if exist "c:\program files (x86)\cmake\bin\cmake.exe" set CMAKEEXE="c:\program files (x86)\cmake\bin\cmake.exe"
"%CMAKEPATH%" -G "Visual Studio 10 2010" -C initcache.cmake ..
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

copy vc2010redist\win32\* Release

cd ..
"c:\program files\7-Zip\7z.exe" a deepcl-win32-%version%.zip dist
if errorlevel 1 exit /B 1

cd ..
echo %version%>latestUnstable.txt

