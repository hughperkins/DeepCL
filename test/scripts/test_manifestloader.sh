#!/bin/bash

# run this from `build` directory, tested on Ubuntu 16.04
# expects mnist at (from `build` directory` ../data/mnist

set -x
set -e

if [[ -d /tmp/testmanifest ]]; then {
    rm -Rf /tmp/testmanifest
    if [[ -f /bin/sync ]]; then {
        sync
    } fi
} fi

mkdir /tmp/testmanifest

source ../dist/bin/activate.sh

MNISTDIR=${MNISTDIR-../data/mnist}
MNISTFILE=$MNISTDIR/train-images-idx3-ubyte
if [ -f $MNISTDIR/train-images.idx3-ubyte ]; then {
    MNISTFILE=$MNISTDIR/train-images.idx3-ubyte
} fi

mnist-to-jpegs $MNISTFILE /tmp/testmanifest 1280

deepcl_train datadir=/tmp/testmanifest trainfile=manifest.txt validatefile=manifest.txt numtrain=1280 numtest=1280 learningrate=0.002 numepochs=3

sed -i -e "s%.*testmanifest/%%g" /tmp/testmanifest/manifest.txt
deepcl_train datadir=/tmp/testmanifest trainfile=manifest.txt validatefile=manifest.txt numtrain=1280 numtest=1280 learningrate=0.002 numepochs=3

head -n 1 /tmp/testmanifest/manifest.txt > /tmp/testmanifest/test.txt
tail -n +2 /tmp/testmanifest/manifest.txt | awk '{print $1}' >> /tmp/testmanifest/test.txt
deepcl_predict writelabels=1 inputfile=/tmp/testmanifest/test.txt outputfile=/tmp/testmanifest/out.txt

head -n 10 /tmp/testmanifest/test.txt > /tmp/testmanifest/test_short.txt
sed -i -e 's/N=1280/N=9/' /tmp/testmanifest/test_short.txt
deepcl_predict writelabels=1 inputfile=/tmp/testmanifest/test_short.txt outputfile=/tmp/testmanifest/out_short.txt

sed -i -e 's/N=9/N=1/' /tmp/testmanifest/test_short.txt
deepcl_predict writelabels=1 inputfile=/tmp/testmanifest/test_short.txt outputfile=/tmp/testmanifest/out_short.txt

