# Copyright Hugh Perkins 2014,2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

import cog

def go(classname, ints = [], floats = []):
    cog.outl( '// generated, using cog:' )
    for thisint in ints:
        cog.outl('int ' + thisint + ' = 0;')
    for thisfloat in floats:
        cog.outl('float ' + thisfloat + ' = 0;')
    for thisint in ints:
        thisintTitlecase = thisint[0].upper() + thisint[1:]
        cog.outl(classname + ' ' + thisintTitlecase + '( int ' + '_' + thisint + ' ) {')
        cog.outl('    this->' + thisint + ' = _' + thisint + ';')
        cog.outl('    return *this;')
        cog.outl('}')
    for thisfloat in floats:
        thisfloatTitlecase = thisfloat[0].upper() + thisfloat[1:]
        cog.outl(classname + ' ' + thisfloatTitlecase + '( float ' + '_' + thisfloat + ' ) {')
        cog.outl('    this->' + thisfloat + ' = _' + thisfloat + ';')
        cog.outl('    return *this;')
        cog.outl('}')

def go1b(classname, ints = [], floats = []):
    cog.outl( '// generated, using cog:' )
    for thisint in ints:
        cog.outl('int ' + thisint + ';')
    for thisfloat in floats:
        cog.outl('float ' + thisfloat + ';')
    cog.outl(classname + '() {')
    for thisint in ints:
        cog.outl('    ' + thisint + ' = 0;')
    for thisfloat in floats:
        cog.outl('    ' + thisfloat + ' = 0;')
    cog.outl('}')
    for thisint in ints:
        thisintTitlecase = thisint[0].upper() + thisint[1:]
        cog.outl(classname + ' ' + thisintTitlecase + '( int ' + '_' + thisint + ' ) {')
        cog.outl('    this->' + thisint + ' = _' + thisint + ';')
        cog.outl('    return *this;')
        cog.outl('}')
    for thisfloat in floats:
        thisfloatTitlecase = thisfloat[0].upper() + thisfloat[1:]
        cog.outl(classname + ' ' + thisfloatTitlecase + '( float ' + '_' + thisfloat + ' ) {')
        cog.outl('    this->' + thisfloat + ' = _' + thisfloat + ';')
        cog.outl('    return *this;')
        cog.outl('}')

def gov2(classname, ints = [], floats = []):
    cog.outl( '// generated, using cog:' )
    for thisint in ints:
        cog.outl('int _' + thisint + ' = 0;')
    for thisfloat in floats:
        cog.outl('float _' + thisfloat + ' = 0;')
    for thisint in ints:
        thisintTitlecase = thisint[0].upper() + thisint[1:]
        cog.outl(classname + ' ' + thisint + '( int ' + '_' + thisint + ' ) {')
        cog.outl('    this->_' + thisint + ' = _' + thisint + ';')
        cog.outl('    return *this;')
        cog.outl('}')
    for thisfloat in floats:
        thisfloatTitlecase = thisfloat[0].upper() + thisfloat[1:]
        cog.outl(classname + ' ' + thisfloat + '( float ' + '_' + thisfloat + ' ) {')
        cog.outl('    this->_' + thisfloat + ' = _' + thisfloat + ';')
        cog.outl('    return *this;')
        cog.outl('}')

def gov3(classname, ints = [], floats = [], strings = []):
    cog.outl( '// generated, using cog:' )
    for thisint in ints:
        cog.outl('int _' + thisint + ';')
    for thisfloat in floats:
        cog.outl('float _' + thisfloat + ';')
    for thisstring in strings:
        cog.outl('std::string _' + thisstring + ';')
    cog.outl(classname + '() {')
    for thisint in ints:
        cog.outl('    _' + thisint + ' = 0;')
    for thisfloat in floats:
        cog.outl('    _' + thisfloat + ' = 0;')
    for thisstring in strings:
        cog.outl('    _' + thisstring + ' = "";')
    cog.outl('}')
    for thisint in ints:
        thisintTitlecase = thisint[0].upper() + thisint[1:]
        cog.outl(classname + ' ' + thisint + '( int ' + '_' + thisint + ' ) {')
        cog.outl('    this->_' + thisint + ' = _' + thisint + ';')
        cog.outl('    return *this;')
        cog.outl('}')
    for thisfloat in floats:
        thisfloatTitlecase = thisfloat[0].upper() + thisfloat[1:]
        cog.outl(classname + ' ' + thisfloat + '( float ' + '_' + thisfloat + ' ) {')
        cog.outl('    this->_' + thisfloat + ' = _' + thisfloat + ';')
        cog.outl('    return *this;')
        cog.outl('}')
    for thisstring in strings:
        thisstringTitlecase = thisstring[0].upper() + thisstring[1:]
        cog.outl(classname + ' ' + thisstring + '( std::string ' + '_' + thisstring + ' ) {')
        cog.outl('    this->_' + thisstring + ' = _' + thisstring + ';')
        cog.outl('    return *this;')
        cog.outl('}')

