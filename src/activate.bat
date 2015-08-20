set "PATH=%~dp0.;%PATH%"
set "CL=%CL% /I%~dp0..\include /I%~dp0..\include\deepcl /I%~dp0..\include\easycl"
set "LIB=%~dp0..\lib;%~dp0..\lib\import %LIB%"
