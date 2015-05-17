# Cog

Cog is an excellent generator framework
* it uses chunks of in-line python to generate in-line code
  * any section starting with `[[[cog` is a code generator section
  * the python to generate code is in the first section, up as far as `]]]`
  * ... and the generated code is in the section section, up to `[[[end]]]`
* it's possible to import python modules, and call those
  * currently, any python module in sub-directory `cog-batteries` can be directly imported

Then, a bunch of modules have been created , in `cog-batteries` directory, to reduce the amount of manual maintenance otherwise necessary.  This means I can do all my coding in gedit and xfce4-terminal, and not feel the need to install Kdevelop et al :-)

## cog_addheaders.py

This is used to fill in the class declarations, in the header file, .h, based on the definitions in the definitions file, .cpp
* prefix any virtual method definitions with `VIRTUAL`
* for static, prefix with `STATIC`
* (New!) if you use `cog_addheaders.addv2()` (cf `.add()`, which is v1), then you can add access specifiers `PUBLIC` or `PROTECTED`
  * by default, in v2, all is `PRIVATE` by default, though you can change this by calling `.addv2(default_access='public')`, if you want

## cog_fluent.py

* Used to construct bean-type classes, whose values can be changed in fluent style, like:

```c++
MyConfig *config = new MyConfig()->setHeight(20)->setWidth(10)->setColor('red');
```

## stringify.py

* Used to write opencl kernels directly into the c++ files
* Compared to using a macro, the advantage is one can see it directly in the code (and I think there is one other important advantage, which eludes my memory right now)
* It uses old-style quotation escaping, rather than new-style c++11 raw strings, for two reasons:
  * originally, since raw-strings are too easy to confuse for maintainable code, when reading the c++ code. (we should be modifying the underlying .cl file, rather than modifying the stringified code, since the stringified code will just be over-written at next build)
  * we're trying to maintain backwards-compatibility with older compilers, that dont support c++11, eg Visual Studio 2008 (for python 2.7), and gcc 4.4

