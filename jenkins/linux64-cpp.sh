pwd
version=$(cat jenkins/version.txt)
rm -Rf build dist
mkdir -p build
cd build
cmake .. || exit 1
make -j 4 install || exit 1
cd ..
tar -cjf deepcl-linux64-${version}.tar.bz2 dist

