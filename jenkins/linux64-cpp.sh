version=$(cat jenkins/version.txt)
echo version ${version}
bash jenkins/linux-cpp.sh
echo version ${version}
tar -cjf deepcl-linux64-${version}.tar.bz2 dist
tar -tf deepcl-linux64-${version}.tar.bz2

