# Coding Guidelines

## Compilers

* On linux, it should be buildable using g++ -std=c++0x
  * this means we can support older compilers, such as gcc 4.4
* On Windows, since we want to be able to build Python wrappers, so we need to support 
Visual Studio 2008 and Visual Studio 2010

## Templating

* not using templating, since it makes writing wrappers for scripting languages harder
* it also causes cryptic debugging messages

## Callbacks, inheritance by client

* callbacks are poorly supported when using from scripting languages, as are
classes that need to be overridden by the client script/code

## Building

* Building is mostly done using cmake.

## cogapp

* [cogapp](http://nedbatchelder.com/code/cog/) generator is used extensively, to accelerate development, reduce the number of manual copy-and-pasting and so on.  Specifically, it's used for:
  * generating header declarations from .cpp definition files
  * generating fluent-style argument classes for certain tests
  * ... and more uses will surely be found :-)
* You need Python installed and available for this to work.  You don't need python just to
build the sources, but if you do have python installed, and you flip the `PYTHON_AVAILABLE`(note: might have changed names recently :-P ) switch in the 
cmake configuration, then a lot of manual editing will no longer be necessary :-)


