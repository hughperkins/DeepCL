# Coding Standards

Note: this page is provisional, and not fixed in stone.  Anything you dislike ->
please raise an issue, or contact the mailinglist.  Thanks :-)

## Compilers

* On linux, it should be buildable using g++ -std=c++0x
  * this means we can support older compilers, such as gcc 4.4, which are still widely in use
* On Windows, since we want to be able to build Python wrappers, so we need to support 
Visual Studio 2008 and Visual Studio 2010
  * in practice everything I've seen that works in Visual Studio 2010 also works in Visual
Studio 2008, and Visual Studio 2010 has a free ide still available
  * therefore current standard on Windows is: must be buildable using Visual Studio 2010 
(although, if something builds on 2010, and not on 2008, these standards might change
slightly at that point ;-) )

## Formatting

* opening-braces on same line, ie:

```c++
if( ... ) {
}
```

## Templating

* please dont use templating, since it's poorly supported by scripting languages,
and makes my brain hurt trying to write wrappers :-)
* it also causes cryptic debugging messages

## Callbacks, inheritance by client

* callbacks are poorly supported when using from scripting languages, as are
classes that need to be overridden by the client script/code
* therefore, prefer not to create an api that needs callbacks or client-side
classes that override DeepCL classes

## Building

* Building is done using cmake where possible.  It is relatively simple to use,
mature, cross-platform, tends to be 'write once, use everywhere', rather than
'write once, debug everywhere' :-)

