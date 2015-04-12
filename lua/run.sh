#!/bin/bash

if [[ x$1 == x ]]; then {
    echo Usage: $0 [scriptname]
    exit 1
} fi

LD_LIBRARY_PATH=../build:. luajit $1

