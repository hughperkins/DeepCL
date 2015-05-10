#!/bin/bash

set -x

tagname=$1
if [[ x${tagname} == x ]]; then {
    echo Usage: $0 [tag name]
    exit 1
} fi

git checkout master || exit 1
git branch -d jenkins-perf || exit 1
git checkout ${tagname} || exit 1
git checkout -b jenkins-perf || exit 1
git push -f --set-upstream origin jenkins-perf || exit 1

