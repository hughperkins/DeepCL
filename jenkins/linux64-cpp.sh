pwd
version=$(cat jenkins/version.txt)
rm -Rf build
mkdir -p build
cd build
cmake ..
make
tar -cjf deepcl-linux64-${version}.tar.bz2 --exclude=CMake* --exclude=CMakeFiles --exclude=cmake* --exclude=Makefile --exclude=*.png --exclude=*.dat *

