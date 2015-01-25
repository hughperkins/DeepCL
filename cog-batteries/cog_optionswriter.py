# Copyright Hugh Perkins 2014,2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

import cog

def write_options( optionsList ):
    cog.outl( '// generated, using cog:' )
    for option in optionsList:
        optionTcase = option[0].upper() + option[1:]
        gOption = 'g' + optionTcase
        cog.outl( 'options += " -D' + gOption + '=" + toString( ' + option + ' );' )

