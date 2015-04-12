# Lua Wrappers

## Concept

These are at the POC stage at the moment.  They're strictly Ubuntu 14.04 only at the moment.
There is no build method for any other environment. Actually, they should probably
work and build on other environments, but you'll need to supply your own build 
mechanism, in place of `build.sh`

## To build

* Firstly read the assumptions and pre-requisites in [build.sh](build.sh)
* Now, simply run:
```
bash build.sh
```
* This will also run the first unit-test, which uses the GenericLoader wrapper to
load mnist labels and images

