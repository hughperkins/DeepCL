#!/bin/bash

# This handles creating rockspecs, rocks, source bundles, and 
# uploading these

# usage:
# ./pack.sh
#     creates a local rock, which we can install locally, to check the source contents are ok
# ./pack.sh upload
#     creates a rockspec, from the rockspec template, and a source tar, and pushes the source tar
#     to git, in ../../DeepCL-ghpages
#     we need to upload the rockspec to luarocks ourselves

# for dev pack, we:
# - point source.url at local dist tarfile
# - create a rock

# for upload pack, we:
# - cp the tar file into ../../DeepCL-ghpages/Downloads/lua
# - git add, commit, push the tar file
# - set source.url to point to ghpages file
# - that's it ... (no need to pack...)

if [[ x$1 == xupload ]]; then {
    echo "Upload selected"
    upload=1
} else {
    echo "Not uploading: creating local rock, add arg 'upload' to upload"
    unset upload
} fi

#apikey=$1
#if [[ x$apikey == x ]]; then {
#    echo 'Usage: $0 [apikey] (get from https://rocks.moonscript.org/settings)'
#    exit 1
#} fi

export package=LuaDeepCL
export packagelower=luadeepcl
export pagesurl=http:\\/\\/hughperkins\\.github\\.io\\/DeepCL\\/Downloads\\/lua
export version=$(head -n 1 version.txt)-1
echo package=${package} version=${version}

set -x
rm -Rf dist/*
mkdir -p dist

cp rockspec dist/${packagelower}-${version}.rockspec
(
    cd dist
    sed -i -e "s/^version = \".*$/version = \"${version}\"/" ${packagelower}-${version}.rockspec || exit 1
    if [[ $upload == 1  ]]; then {
        sed -i -e "s/^ *url = \".*$/    url = \"${pagesurl}\/${package}-${version}.tar.gz\"/" ${packagelower}-${version}.rockspec || exit 1
    } fi
)

mkdir -p dist/${package}-${version}
mkdir -p dist/${package}-${version}/pkgsrc

rsync -av ../src/ dist/${package}-${version}/pkgsrc/src/
rsync -av ../qlearning/ dist/${package}-${version}/pkgsrc/qlearning/
rsync -av ../EasyCL/ dist/${package}-${version}/pkgsrc/EasyCL/

cp *.lua *.i CMakeLists.txt *.cxx dist/${package}-${version}/
rsync -av thirdparty/ dist/${package}-${version}/thirdparty/

touch dist/${package}-${version}/inpkg.flag
(cd dist; tar czvpf ${package}-${version}.tar.gz ${package}-${version}/)

if [[ $upload == 1  ]]; then {
    cp dist/${package}-${version}.tar.gz ../../DeepCL-ghpages/Downloads/lua
    (
        cd ../../DeepCL-ghpages
        git add Downloads/lua/${package}-${version}.tar.gz;
        git commit -m "lua rock upload"
        git push
    )
} else  {
    cd dist
    luarocks pack ${packagelower}-${version}.rockspec 
} fi

# I abandoned upload directly; cant remember why... oh yes, it needs a json
# library, but doesnt say how to obtain/install one
# luarocks upload --api-key $apikey dist/${packagelower}-${version}.rockspec


