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

