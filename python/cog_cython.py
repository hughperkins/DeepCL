# Copyright Hugh Perkins 2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.

"""
functions to help wrap C++ callback classes in Cython, and more
There are three parts to wrapping C++ callback classes:

- in C++, you need to override the C++-side abstract class
  => cpp_write_proxy_class

- in the pxd, you need to declare the C++ proxy class
  => pxd_write_proxy_class

- in the .pyx, you need to write a wrapper class, that can be
  overridden in the python files
  => pyx_write_overrideable_class

in all cases, you need to provide a 'defs' file, which is a python
file with a list of method definitions, provided as tuples like:
defs.append(('act', 'float', [('int','index')]))
here: - act is the name of the method
      - float is the return type
      - there is one parameter 'index', of type 'int'
"""

import cog


def upperFirst(word):
    """
    helper method to capitalize the first letter of word
    """
    word = word[0].upper() + word[1:]
    return word


def cpp_write_proxy_class(proxy_name, parent_name, defs):
    """
    use to create a c++ class that inherits from a (possibly abstract) c++ class
    and handles the c++ side of receiving callback functions into cython,
    and calling these appropriately
    """

    cog.outl('// generated using cog (as far as the [[end]] bit:')
    cog.outl('class ' + proxy_name + ' : public ' + parent_name + ' {')
    cog.outl('public:')
    cog.outl('    void *pyObject;')
    cog.outl('')
    cog.outl('    ' + proxy_name + '(void *pyObject) :')
    cog.outl('        pyObject(pyObject) {')
    cog.outl('    }')
    cog.outl('')

    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        cog.out('    typedef ' + returnType + '(*' + name + 'Def)(')
        for parameter in parameters:
            (ptype, pname) = parameter
            cog.out(ptype + ' ' + pname + ',')
        cog.outl(' void *pyObject);')
    cog.outl('')

    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        cog.outl('    ' + name + 'Def c' + upperFirst(name) + ';')
    cog.outl('')

    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        cog.outl(
            '    void set' + upperFirst(name) + ' (' +
            name + 'Def c' + upperFirst(name) + ') {')
        cog.outl(
            '        this->c' + upperFirst(name) +
            ' = c' + upperFirst(name) + ';')
        cog.outl('    }')
    cog.outl('')

    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        cog.out('    virtual ' + returnType + ' ' + name + '(')
        isFirstParam = True
        for param in parameters:
            (ptype, pname) = param
            if not isFirstParam:
                cog.out(', ')
            cog.out(ptype + ' ' + pname)
            isFirstParam = False
        cog.outl(') {')
        cog.out('        ')
        if returnType != 'void':
            cog.out('return ')
        cog.out('c' + upperFirst(name) + '(')
        for param in parameters:
            (ptype, pname) = param
            cog.out(pname + ', ')
        cog.outl('pyObject);')
        cog.outl('    }')
    cog.outl('};')


def pxd_write_proxy_class(proxy_name, defs):
    """
    writes the pxd declaration of the same class that was created using
    'cpp_write_proxy_class' for C++ above.
    This should be used inside 'cdef extern from "somefile.h":' section
    """

    cog.outl('# generated using cog (as far as the [[end]] bit:')
    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        cog.out(
            'ctypedef ' + returnType +
            '(*' + proxy_name + '_' + name + 'Def)(')
        for parameter in parameters:
            (ptype, pname) = parameter
            cog.out(ptype + ' ' + pname + ',')
        cog.outl(' void *pyObject)')

    cog.outl('cdef cppclass ' + proxy_name + ':')
    cog.outl('    ' + proxy_name + '(void *pyObject)')
    cog.outl('')
    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        cog.outl(
            '    void set' + upperFirst(name) +
            ' (' + proxy_name + '_' + name + 'Def c' + upperFirst(name) + ')')


def pyx_write_overrideable_class(
        pxd_module, pxd_class, pyx_class, defs, skip_names):
    """
    writes the python class in the pyx file that the .py modules
    can override, and receives callbacks from
    any method names in skip_names will be skipped, and you can write them
    manually before/after the cog block
    """

    cog.outl('# generated using cog (as far as the [[end]] bit:')
    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        if name not in skip_names:
            cog.out('cdef ' + returnType + ' ' + pyx_class + '_' + name + '(')
            for (ptype, pname) in parameters:
                cog.out(ptype + ' ' + pname + ', ')
                isFirst = False
            cog.outl(' void *pyObject):')
            cog.out('    ')
            if returnType != 'void':
                cog.out('return ')
            cog.out('(<object>pyObject).' + name + '(')
            isFirst = True
            for (ptype, pname) in parameters:
                if not isFirst:
                    cog.out(', ')
                cog.out(pname)
                isFirst = False
            cog.outl(')')
            cog.outl('')
    cog.outl('cdef class ' + pyx_class + ':')
    cog.outl('    cdef ' + pxd_module + '.' + pxd_class + ' *thisptr')
    cog.outl('    def __cinit__(self):')
    cog.outl(
        '        self.thisptr = new ' +
        pxd_module + '.' + pxd_class + '(<void *>self)')
    cog.outl('')
    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        cog.outl(
            '        self.thisptr.set' + upperFirst(name) +
            '(' + pyx_class + '_' + name + ')')
    cog.outl('')
    for thisdef in defs:
        (name, returnType, parameters) = thisdef
        if name in skip_names:
            continue
        cog.out('    def ' + name + '(self')
        for (ptype, pname) in parameters:
            cog.out(', ' + pname)
        cog.outl('):')
        cog.outl(
            '        raise Exception("Method needs to be overridden: ' +
            pyx_class + '.' + name + '()")')
        cog.outl('')
