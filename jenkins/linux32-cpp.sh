pwd
version=$(cat jenkins/version.txt)
rm -Rf build
mkdir -p build
cd build
schroot -c trusty_i386 -- cmake -D BUILD_PYTHON_WRAPPERS:BOOL=OFF -D BUILD_LUA_WRAPPERS:BOOL=OFF ..
schroot -c trusty_i386 -- make
tar -cjf deepcl-linux32-${version}.tar.bz2 --exclude=CMake* --exclude=CMakeFiles --exclude=cmake* --exclude=Makefile --exclude=*.png --exclude=*.dat *
    
