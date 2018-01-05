<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Specific dataset examples](#specific-dataset-examples)
  - [NORB](#norb)
  - [MNIST](#mnist)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Specific dataset examples

## NORB

* Download the data files from [NORB datafiles](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/), and place in `data/norb`, decompressed, ie (g)unzipped
* Pre-process, to shuffle the training samples, and draw 1000 testing samples:
```bash
./prepare-norb ../data/norb
```
* Run training, eg, based on LeCun's lenet-7:
```bash
deepcl_train netdef=8C5-MP4-24C6-MP3-80C6-5N learningrate=0.00001 dataset=norb
```
* On an Amazon AWS GPU instance, which has an NVidia GRID K520 GPU, this has epoch time of 76 seconds, and reaches test accuracy of around 91.7% after around 200 epochs (train accuracy 99.996%!)

## MNIST

* You can download the MNIST data from [MNIST database](http://yann.lecun.com/exdb/mnist/) , and place in the `data\mnist` directory, (g)unzipped.
* Run as per the [convnetjs MNIST demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html) architecture as follows:
```bash
deepcl_train netdef=8c5{padzeros}-mp2-16c5{padzeros}-mp3-10n learningrate=0.002 dataset=mnist
```
* On an Amazon AWS GPU instance, epoch time is about 13.8 seconds, giving about 98.7% test accuracy, after 12 epochs
* Actually, I think the following gives slightly better test accuracy, about 99.0%, using 17.2seconds per epoch:
```bash
deepcl_train netdef=8c5{padzeros}-mp2-16c5{padzeros}-mp3-150n-10n learningrate=0.002 dataset=mnist
```

