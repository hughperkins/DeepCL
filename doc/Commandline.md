<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Commandline usage](#commandline-usage)
  - [Additional net-def options](#additional-net-def-options)
  - [Repeated layers](#repeated-layers)
  - [Additional layer types](#additional-layer-types)
    - [Random patches](#random-patches)
    - [Random translations](#random-translations)
  - [Multi-column deep neural network "MultiNet"](#multi-column-deep-neural-network-multinet)
  - [Pre-processing](#pre-processing)
  - [File types](#file-types)
  - [Weight persistence](#weight-persistence)
  - [Command-line options](#command-line-options)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Commandline usage

* Syntax is based on that specified in Ciresan et al's [Multi-column Deep Neural Networks for Image Classification](http://arxiv.org/pdf/1202.2745.pdf), section 3, first paragraph:
  * network is defined by a string like: `100C5-MP2-100C5-MP2-100C4-MP2-300N-100N-6N`
  * `100C5` means: a convolutional layer, with 100 filters, each 5x5
  * `MP2` means a max-pooling layer, over non-overlapping regions of 2x2
  * `300N` means a fully connected layer with 300 hidden units
* Thus, you can do, for example:
```bash
./deepclrun netdef=8c5-mp2-16c5-mp3-10n learningrate=0.002 dataset=mnist
```
... in order to learn mnist, using the same neural net architecture as used in the [convnetjs mnist demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)
* Similarly, you can learn NORB, using approximately the architecture specified in [lecun-04](http://yann.lecun.com/exdb/publis/pdf/lecun-04.pdf), by doing:
```bash
./deepclrun netdef=8C5-MP4-24C6-MP3-80C6-5N learningrate=0.0001 dataset=norb
```
* Or, you can train NORB using the very deep, broad architecture specified by Ciresan et al in [Flexible, High Performance Convolutional Neural Networks for Image Classification](http://ijcai.org/papers11/Papers/IJCAI11-210.pdf):
```bash
./deepclrun netdef=MP3-300C6-MP2-500C4-MP4-500N-5N learningrate=0.0001 dataset=norb
```

## Additional net-def options

* You can add additional options in `{}` brackets after each layer, eg:
```bash
./deepclrun netdef=8c5{tanh}-mp2-16c5{tanh}-mp3-10n learningrate=0.002 dataset=mnist
```
* Options currently available:
  * For convolution layers:
    * `sigmoid`
    * `tanh`
    * `scaledtanh` (ie, 1.7159f * tanh( 0.66667f * x ) )
    * `linear`
    * `relu` (default)
    * `padzeros` (or simply `z`)
* can be combined, comma-separated (no spaces), eg:
```bash
./deepclrun netdef=8c5{tanh,z}-mp2-16c5{tanh,z}-mp3-10n learningrate=0.002 dataset=mnist
```

## Repeated layers

* simply prefix a layer with eg `3*` to repeat it.  `3*` will repeat the layer 3 times, and similar for other numbers, eg:
```
./deepclrun netdef=6*32c5{z}-500n-361n learningrate=0.0001 dataset=kgsgoall
```
... will create 6 convolutional layers of 32 5x5 filters each.
* you can also use parentheses `(...)` to repeat multiple layers, eg:
```
./deepclrun netdef=3*(32c5{z}-mp2)-150n-10n
```
... will be expanded to:
```
./deepclrun netdef=32c5{z}-mp2-32c5{z}-mp2-32c5{z}-mp2-150n-10n
```


## Additional layer types

### Random patches

* `RP24` means a random patch layer, which will cut a 24x24 patch from a random position in each incoming image, and send that to its output
* during testing, the patch will be cut from the centre of each image
* can reduce over-training, and thus give better test accuracies
* image size output by this layer equals the the patch size
* eg you can try, on MNIST, `netdef=rp24-8c5{padzeros}-mp2-16c5{padzeros}-mp3-150n-10n`
* example paper using this approach: [ImageNet Classification with Deep Convolutional Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

### Random translations

* `RT2` means a random translations layer, which will translate the image randomly during training, up to 2 pixels, in either direction, along both axes
* Can specify any non-negative integer, less than the image size
* Image size is unchanged by this layer
* During testing, no translation is done
* eg you can try, on MNIST, `netdef=rt2-8c5{padzeros}-mp2-16c5{padzeros}-mp3-150n-10n`
* example paper using this approach: [Flexible, High Performance Convolutional Neural Networks for Image Classification](http://ijcai.org/papers11/Papers/IJCAI11-210.pdf)

## Multi-column deep neural network "MultiNet"

* You can train several neural networks at the same time, and predict using the average output across all of them using the `multinet` option
* Simply add eg `multinet=3` in the commandline, to train across 3 nets in parallel, or put a number of your choice

## File types

* Using the new `GenericLoader.cpp`, it's possible to automatically detect various filetypes
* When specifying a training or validation file, if there is both a labels and an images file, then specify the images file, and the labels file will be detected automatically
* Currently, the following filetypes are supported:
  * Norb .mat format as specified at [NORB-small dataset](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/)
  * MNIST format, as specified at [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
  * kgs go v2 format, [https://github.com/hughperkins/kgsgo-dataset-preprocessor](https://github.com/hughperkins/kgsgo-dataset-preprocessor)
* For other formats, as long as the format has a recognizable header section, there's no particular reason why it couldnt be added

## Weight persistence

* If we're going to train for hours or days, we probably want to make sure that if our process gets interrupted, we don't lose our training so far
* By default, weights will be written to `weights.dat`, after each epoch
* If you specify option `loadweights=1`, the weights will be loadeded at the start
* You can change the weights file with option `weightsfile=somefilename.dat`
* If you specify option `loadweights=1`:
  * the `netdef`, the `datadir`, and the `trainfile` will be compared to that used to generate the current weights file: if any of them are different, then DeepCL will refuse to carry on, so that you don't overwrite the weights file inadvertently
    * If you need something like, DeepCL prompts you, showing the differences, and asks if you want to continue, then please raise an issue, and I will add this in
  * Note that the epoch number will continue from the weights.dat file, so make sure to increase `numepochs` appropriately, otherwise DeepCL will start, load the weights file, and then exit again, since all epochs have been finished :-P

## Command-line options

| Option | Description |
|----|----|
| dataset=norb | sets datadir, trainfile and validatefile according to one of several dataset profiles.  Current choices: mnist, norb, cifar10, kgsgo, kgsgoall |
| datadir=../data/mnist | path to data files |
| trainfile=train-dat.mat | name of training data file, the one with the images in.  Note that the labels file will be determined automatically, based on the data filename and type, eg in this case `train-cat.mat` |
| validationfile=validate-dat.mat | name of the validation data file, the one with the images in.  Note that the labels file will be determined automatically, based on the data filename and type, eg in this case `validate-cat.mat` |
| numtrain=1000 | only uses the first 1000 training samples |
| numtest=1000 | only uses the first 1000 testing samples |
| netdef=100c5-10n | provide the network definition, as documented in [Commandline usage](#commandline-usage]) above |
| learningrate=0.0001 | specify learning rate |
| anneallearningrate=0.95 | multiply learning rate by this after each epoch, as described in [Ciresan et al](http://ijcai.org/papers11/Papers/IJCAI11-210.pdf) |
| numepochs=20 | train for this many epochs |
| batchsize=128 | size of each mini-batch.  Too big, and the learning rate will need to be reduced.  Too small, and performance will decrease.  128 might be a reasonable compromise |
| normalization=maxmin | can choose maxmin or stddev.  Default is stddev |
| normalizationnumstds=2 | how many standard deviations from mean should be +1/-1?  Default is 2 |
| normalizationexamples=50000 | how many examples to read, to determine normalization values |
| multinet=3 | train 3 networks at the same time, and predict using average output from all 3, can put any integer greater than 1 |
| loadondemand=1 | Load the file in chunks, as learning proceeds, to reduce memory requirements. Default 0 |
| filebatchsize=50 | When loadondemand=1, load this many batches at a time.  Numbers larger than 1 increase efficiency of disk reads, speeding up learning, but use up more memory |
| weightsfile=weights.dat | file to store weights in, after each epoch.  If blank, then weights not stored |
| loadweights=1 | load weights at start, from weightsfile.  Current training config, ie netdef and trainingfile, should match that used to create the weightsfile.  Note that epoch number will continue from file, so make sure to increase numepochs sufficiently |



