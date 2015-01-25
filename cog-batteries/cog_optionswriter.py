import cog

def write_options( optionsList ):
    cog.outl( '// generated, using cog:' )
    for option in optionsList:
        optionTcase = option[0].upper() + option[1:]
        gOption = 'g' + optionTcase
        cog.outl( 'options += " -D' + gOption + '=" + toString( ' + option + ' );' )

