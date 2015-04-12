# Lua Wrappers

## Concept

These are at the POC stage at the moment.  They're strictly Ubuntu 14.04 only at the moment.
There is no build method for any other environment. Actually, they should probably
work and build on other environments, but you'll need to supply your own build 
mechanism, in place of [build.sh](build.sh)

## Demo

The following classes are available from lua now:
* NeuralNet
* NetLearner
* GenericLoader
* NetdefToNet
* NormalizationLayerMaker

This is enough to load data, create a net, and train it :-)

For a demo, you can have a look at the method `test_basic` in [test_lua.lua](test_lua.lua)

## To build

* Firstly read the assumptions and pre-requisites in [build.sh](build.sh)
* Now, simply run:
```
bash build.sh
```
* This will also run the first unit-test, which uses the GenericLoader wrapper to
load mnist labels and images

