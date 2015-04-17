#!/bin/bash

if [[ x$1 == x ]]; then {
    echo Usage: $0 [scriptname]
    exit 1
} fi

LUA_CPATH=build/?.so luajit $1

