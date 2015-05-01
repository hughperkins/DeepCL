#!/bin/bash

if [[ x$1 == x ]]; then {
    echo Usage: $0 [scriptname]
    exit 1
} fi

echo 1=$1
echo 2=$2
echo 3=$3
LUA_CPATH=build/?.so LUA_PATH=thirdparty/luaunit/?.lua gdb luajit -ex "run $1 $2 $3 $4 $5 $6 $7 $8 $9"

