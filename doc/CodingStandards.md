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

