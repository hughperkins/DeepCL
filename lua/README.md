# Lua Wrappers

## Concept

Lua wrappers are available.  You'll need to build them yourself for now though.
For linux, Ubuntu 14.04, there is a batch file to run the build.

## Demo

* For a demo of high-level functions, to create a network and train it, you can have a look at the method `test_basic` in [test_lua.lua](test_lua.lua)
* For a demo of constructing layers by hand, and handling low-level propagating, batch by batch, you can look at the method `test_lowlevel`, in the same module, ie in [test_lua.lua](test_lua.lua).
* For a demo of q-learning, you can look at [test_qlearning.lua](test_qlearning.lua)

## To build

* If you're on linux, firstly read the assumptions and pre-requisites in [build.sh](build.sh)
* Now run:
```
bash build.sh
```

## To run

According to which module you want to run, you can run one of:
```bash
./run.sh test_lua.lua
```
or:
```bash
./run.sh test_qlearning.lua
```

## Unit-testing

* The source-code includes the thirdparty lua unit-test tool luaunit.  [test_lua.lua](test_lua.lua)
creates some first initial unit tests

