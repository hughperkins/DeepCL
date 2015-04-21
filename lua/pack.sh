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
mkdir -p dist/${packagelower}-${version}
mkdir -p dist/${packagelower}-${version}/pkgsrc
rsync -av ../src/ dist/${packagelower}-${version}/pkgsrc/src/
cp *.lua *.i CMakeLists.txt *.cxx dist/${packagelower}-${version}/
touch dist/${packagelower}-${version}/inpkg.flag
(cd dist; tar czvpf ${package}-${version}.tar.gz ${packagelower}-${version}/)

cp rockspec ${packagelower}-${version}.rockspec
sed -i -e "s/^version = \".*$/version = \"${version}\"/" ${packagelower}-${version}.rockspec

luarocks pack ${packagelower}-${version}.rockspec


