#!/bin/bash

# this can be run on linux, to create the rock
# at least, that is the goal

export package=LuaDeepCL
export packagelower=luadeepcl
export version=$(head -n 1 version.txt)
echo packagelower=${packagelower} version=${version}

rm *.rock
rm *.rockspec

rm -Rf dist/*
mkdir -p dist
mkdir -p dist/${package}-${version}
mkdir -p dist/${package}-${version}/pkgsrc

rsync -av ../src/ dist/${package}-${version}/pkgsrc/src/
rsync -av ../qlearning/ dist/${package}-${version}/pkgsrc/qlearning/
rsync -av ../OpenCLHelper/ dist/${package}-${version}/pkgsrc/OpenCLHelper/

cp *.lua *.i CMakeLists.txt *.cxx dist/${package}-${version}/
rsync -av thirdparty/ dist/${package}-${version}/thirdparty/

touch dist/${package}-${version}/inpkg.flag
(cd dist; tar czvpf ${package}-${version}.tar.gz ${package}-${version}/)

cp rockspec ${packagelower}-${version}.rockspec
sed -i -e "s/^version = \".*$/version = \"${version}\"/" ${packagelower}-${version}.rockspec

luarocks pack ${packagelower}-${version}.rockspec


