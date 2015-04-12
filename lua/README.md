# Lua Wrappers

## Concept

These are at the POC stage at the moment.  They're strictly Ubuntu 14.04 only at the moment.
There is no build method for any other environment. Actually, they should probably
work and build on other environments, but you'll need to supply your own build 
mechanism, in place of [build.sh](build.sh)

## Demo

* For a demo of high-level functions, to create a network and train it, you can have a look at the method `test_basic` in [test_lua.lua](test_lua.lua)
* For a demo of constructing layers by hand, and handling low-level propagating, batch by batch, you can look at the method `test_lowlevel`, in the same module, ie in [test_lua.lua](test_lua.lua).
* For a demo of q-learning, you can look at [test_qlearning.lua](test_qlearning.lua)

## To build

* Firstly read the assumptions and pre-requisites in [build.sh](build.sh)
* Now, simply run:
```
bash build.sh
```
* This will also run the first unit-test, which uses the GenericLoader wrapper to
load mnist labels and images

## To run

According to which module you want to run, you can run one of:
```bash
./run.sh test_lua.lua
```
or:
```bash
./run.sh test_qlearning.lua
```


