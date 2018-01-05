version=$(cat jenkins/version.txt)
export PATH
echo version ${version}
schroot -c trusty_i386 bash jenkins/linux-cpp.sh || exit 1
echo version ${version}
tar -cjf deepcl-linux32-${version}.tar.bz2 dist
tar -tf deepcl-linux32-${version}.tar.bz2

