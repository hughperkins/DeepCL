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

## Training

Use `train` to run training (`deepclrun` in v5.8.3 and below).

* Syntax is based on that specified in Ciresan et al's [Multi-column Deep Neural Networks for Image Classification](http://arxiv.org/pdf/1202.2745.pdf), section 3, first paragraph:
  * network is defined by a string like: `100C5-MP2-100C5-MP2-100C4-MP2-300N-100N-6N`
  * `100c5` means: a convolutional layer, with 100 filters, each 5x5
  * adding `z` to a convolutional layer makes it zero-padded, eg `8c5z` is: a convolutional layer, with 8 filters, each 5x5, zero-padded
  * `mp2` means a max-pooling layer, over non-overlapping regions of 2x2
  * `300n` means a fully connected layer with 300 hidden units
  * `relu` means a relu layer
  * `tanh` means a tanh layer
* Thus, you can do, for example:
```bash
deepcl_train netdef=8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n learningrate=0.002 dataset=mnist
```
... in order to learn mnist, using the same neural net architecture as used in the [convnetjs mnist demo](http://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)
* Similarly, you can learn NORB, using approximately the architecture specified in [lecun-04](http://yann.lecun.com/exdb/publis/pdf/lecun-04.pdf), by doing:
```bash
deepcl_train netdef=8c5-relu-mp4-24c6-relu-mp3-80c6-relu-5n learningrate=0.0001 dataset=norb
```
* Or, you can train NORB using the very deep, broad architecture specified by Ciresan et al in [Flexible, High Performance Convolutional Neural Networks for Image Classification](http://ijcai.org/papers11/Papers/IJCAI11-210.pdf):
```bash
deepcl_train netdef=MP3-300C6-RELU-MP2-500C4-RELU-MP4-500N-TANH-5N learningrate=0.0001 dataset=norb
```

### Convolutional

* eg `-32c5` is a convolutional layer with 32 filters of 5x5
* `-32c5z` is a convolutional layer with zero-padding, of 32 filters of 5x5

### Fully-connected

* eg `-150n` is a fully connected layer, with 150 neurons.

### Max-pooling

* Eg `-mp3` will add a max-pooling layer, over 3x3 non-overlapping regions.  The number is the size of the regions, and can be modified

### Dropout layers

* Simply add `-drop` into the netdef string
  * this will use a dropout ratio of 0.5

### Activation layers

* Simply add any of the following into the netdef string:
  * `-tanh`
  * `-sigmoid`
  * `-relu`
  * `-elu`

#### Random patches

* `RP24` means a random patch layer, which will cut a 24x24 patch from a random position in each incoming image, and send that to its output
* during testing, the patch will be cut from the centre of each image

#### Random translations

* `RT2` means a random translations layer, which will translate the image randomly during training, up to 2 pixels, in either direction, along both axes
* Can specify any non-negative integer, less than the image size
* During testing, no translation is done

### Multi-column deep neural network "MultiNet"

* You can train several neural networks at the same time, and predict using the average output across all of them using the `multinet` option
* Simply add eg `multinet=3` in the commandline, to train across 3 nets in parallel, or put a number of your choice

### Repeated layers

* simply prefix a layer with eg `3*` to repeat it.  `3*` will repeat the layer 3 times, and similar for other numbers, eg:
```
deepcl_train netdef=6*(32c5z-relu)-500n-361n learningrate=0.0001 dataset=kgsgoall
```
... will create 6 convolutional layers of 32 5x5 filters each.
* you can also use parentheses `(...)` to repeat multiple layers, eg:
```
deepcl_train netdef=3*(32c5z-relu-mp2)-150n-10n
```
... will be expanded to:
```
deepcl_train netdef=32c5z-relu-mp2-32c5z-relu-mp2-32c5z-relu-mp2-150n-10n
```

### File types

* Simply pass in the filename of the data file with the images in
* Filetype will be detected automatically
* See [Loaders](Loaders.md) for information on available loaders

### Weight persistence

* By default, weights will be written to `weights.dat`, after each epoch
  * You can add option `writeweightsinterval=5` to write weights every 5 minutes, even if the epoch hasnt finished yet.  Just replace `5` with the number of minutes between each write
* If you specify option `loadweights=1`, the weights will be loadeded at the start
* You can change the weights filepath with option eg `weightsfile=somefilename.dat`
* If you specify option `loadweights=1`, the `netdef` will be compared to that used to generate the current weights file: if it is different, then DeepCL will ask you if you're sure you want to continue, to avoid corrupting the weights file
* Epoch number, batch number, batch loss, and batch numcorrect will all be loaded from where they left off, from the weights file, so you can freely stop and start training, without losing the training
  * be sure to use the `writeweightsinterval=5` option if you are going to stop/start often, with long epochs, to avoid losing hours/days of training!

### Command-line options

| Option | Description |
|----|----|
| gpuindex=1 | choose which gpu device to use.  Default -1 means first gpu, or else cpu.  Otherwise, gpu index from 0 |
| dataset=norb | sets datadir, trainfile and validatefile according to one of several dataset profiles.  Current choices: mnist, norb, cifar10, kgsgo, kgsgoall |
| datadir=../data/mnist | path to data files |
| trainfile=train-dat.mat | name of training data file, the one with the images in.  Note that the labels file will be determined automatically, based on the data filename and type, eg in this case `train-cat.mat` |
| validationfile=validate-dat.mat | name of the validation data file, the one with the images in.  Note that the labels file will be determined automatically, based on the data filename and type, eg in this case `validate-cat.mat` |
| numtrain=1000 | only uses the first 1000 training samples |
| numtest=1000 | only uses the first 1000 testing samples |
| netdef=100c5-10n | provide the network definition, as documented in [Commandline usage](#commandline-usage]) above |
| weightsinitializer=uniform | choose weight initializer.  valid choices: original, uniform (default: original) |
| initialweights=10 | set size of initial weights, sampled uniformally from range +/- initialweights divided by fanin. used by uniform initializer (default: 1.0) |
| trainer=sgd | choose trainer.  valid choices are sgd, anneal, nesterov, adagrad, or rmsprop.  (default: sgd) |
| learningrate=0.0001 | specify learning rate. works with any trainer, except adadelta |
| momentum=0.1 | specify momentum (default: 0). works with sgd and nesterov trainers |
| rho=0.9 | rho decay, from equation 1 of adadelta paper, http://arxiv.org/pdf/1212.5701v1.pdf (default: 0.9) |
| weightdecay=0.001 | weight decay, 0 means no decay, 1 means complete decay (default:0). works with sgd trainer |
| anneal=0.95 | anneal learning.  1 means no annealing. 0 means learningrate is 0 (default:1). works with anneal trainer |
| numepochs=20 | train for this many epochs |
| batchsize=128 | size of each mini-batch.  Too big, and the learning rate will need to be reduced.  Too small, and performance will decrease.  128 might be a reasonable compromise |
| normalization=maxmin | can choose maxmin or stddev.  Default is stddev |
| normalizationnumstds=2 | how many standard deviations from mean should be +1/-1?  Default is 2 |
| normalizationexamples=50000 | how many examples to read, to determine normalization values |
| multinet=3 | train 3 networks at the same time, and predict using average output from all 3, can put any integer greater than 1 |
| loadondemand=1 | Load the file in chunks, as learning proceeds, to reduce memory requirements. Default 0 |
| filebatchsize=50 | When loadondemand=1, load this many batches at a time.  Numbers larger than 1 increase efficiency of disk reads, speeding up learning, but use up more memory |
| weightsfile=weights.dat | file to store weights in, after each epoch.  If blank, then weights not stored |
| writeweightsinterval=5 | write the weights to file every 5 minutes of training, even if epoch hasnt finished yet.  Default is 0, ie only write weights after each epoch |
| loadweights=1 | load weights at start, from weightsfile.  Current training config, ie netdef and trainingfile, should match that used to create the weightsfile.  Note that epoch number will continue from file, so make sure to increase numepochs sufficiently |

## Prediction

Use `deepcl_predict` to run prediction  (`deepclexec` in v5.8.3 and below)


