version=$(cat jenkins/version.txt)
export PATH
echo version ${version}
bash jenkins/linux-cpp.sh || exit 1
echo version ${version}
tar -cjf deepcl-linux64-${version}.tar.bz2 dist
tar -tf deepcl-linux64-${version}.tar.bz2

