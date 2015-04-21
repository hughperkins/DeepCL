#!/bin/bash

# this can be run on linux, to create the rock
# at least, that is the goal

export package=DeepCL
export version=$(head -n 1 version.txt)
echo package=${package} version=${version}

rm *.rock
rm *.rockspec

rm -Rf dist/*
mkdir -p dist
mkdir -p dist/${package}-${version}
mkdir -p dist/${package}-${version}/pkgsrc
rsync -av ../src/ dist/${package}-${version}/pkgsrc/src/
cp *.lua *.i CMakeLists.txt *.cxx dist/${package}-${version}/
touch dist/${package}-${version}/inpkg.flag
(cd dist; tar czvpf ${package}-${version}.tar.gz ${package}-${version}/)

cp rockspec ${package}-${version}.rockspec
sed -i -e "s/^version = \".*$/version = \"${version}\"/" ${package}-${version}.rockspec

luarocks pack ${package}-${version}.rockspec


