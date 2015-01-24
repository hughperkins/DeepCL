import cog

def go(classname, ints = [], floats = []):
    for thisint in ints:
        cog.outl('int ' + thisint + ' = 0;')
    for thisfloat in floats:
        cog.outl('float ' + thisfloat + ' = 0;')
    cog.outl('')
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

