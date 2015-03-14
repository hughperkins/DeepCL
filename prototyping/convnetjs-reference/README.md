# convnetjs reference implementation

I'm using (since 13th March 2015) [convnetjs](https://github.com/karpathy/convnetjs) as a reference implementation, to double-check
my calculations, since convnetjs:

 - is straightforward to read
 - widely used/forked, therefore probably correct
 - easy to run, doesnt need a gpu etc
 - I like nodejs :-)

## Pre-requisites

* need latest version of nodejs, I downloaded from [node v0.12.0](http://nodejs.org/dist/v0.12.0/node-v0.12.0-linux-x64.tar.gz), I'm running Ubuntu 14.04 64-bit
* the associated nodejs `bin` directory should be in the path

## To install/configure:

```bash
npm install
```

## To run:

```bash
npm start
```

Possible args, eg:
```bash
npm start numtrain=4 numepochs=4
```

## To convert png into norb .mat format

* This is used to ensure we are using the exact same data when we run ClConvolve, as when we run convnetjs
* First, run the convnetjs implementation above, which will download the data files
* Then, run:
```bash
node pngtomat.js
```
* This will generate the following files, in the `data` subdirectory:
  * `mnist12k-dat.mat`
  * `mnist12k-cat.mat`
* You can use these with ClConvolve by setting `datadir=` to point to the `data` subdirectory, and specifying `trainfile=mnist12k-dat.mat`

## How to run clconvolve, to get comparable results

* Use [clconvolve-fixedweights.cpp](prototyping/clconvolve-fixedweights.cpp]
* eg:
```bash
./clconvolve-fixedweights datadir=../prototyping/convnetjs-reference/data trainfile=mnist12k-dat.mat validatefile=mnist12k-dat.mat 'netdef=10n{linear}' numtrain=1 batchsize=1 numepochs=1 learningrate=0.4 normalizationexamples=1
```

