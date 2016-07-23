#!/bin/bash

# run this from `build` directory, tested on Ubuntu 16.04
# expects mnist at (from `build` directory` ../data/mnist

if [[ -d /tmp/testmanifest ]]; then {
    rm -Rf /tmp/testmanifest
} fi

mkdir /tmp/testmanifest

./mnist-to-jpegs ../data/mnist/train-images-idx3-ubyte /tmp/testmanifest 1280
./deepcl_train datadir=/tmp/testmanifest trainfile=manifest.txt validatefile=manifest.txt numtrain=1280 numtest=1280 learningrate=0.002 numepochs=3

sed -i -e "s%/tmp/testmanifest/%%g" /tmp/testmanifest/manifest.txt
./deepcl_train datadir=/tmp/testmanifest trainfile=manifest.txt validatefile=manifest.txt numtrain=1280 numtest=1280 learningrate=0.002 numepochs=3

