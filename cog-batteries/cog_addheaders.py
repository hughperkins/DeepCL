# Copyright Hugh Perkins 2014,2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.
# 
# simply import this module in your header file like:
#
#    // [[[cog
#    // import cog_addheaders
#    // cog_addheaders.add()
#    // ]]]
#    // [[[end]]]
#
# ... and run cog on the header file, to generate the header declarations

import cog

def add( classname = ''):
#    debug = open('debug.txt', 'a' )
#    debug.write( 'foo\n')
#    debug.write( 'infile [' + cog.inFile + ']\n' )

    infile = cog.inFile
    splitinfile = infile.replace('\\','/').split('/')
    infilename = splitinfile[ len(splitinfile) - 1 ]
    if classname == '':
        classname = infilename.replace('.h','')
    cppfile = infile.replace('.h','.cpp')
    # cog.outl( '// classname: ' + classname )
    # cog.outl( '// cppfile: ' + infilename.replace('.h','.cpp' ) )
    f = open( cppfile, 'r')
    in_multiline_comment = False
    in_header = False;
    line = f.readline()
    cog.outl( '// generated, using cog:' )
    while( line != '' ):
       # cog.outl(line)
       if( line.strip().find("/*") >= 0 ):
           in_multiline_comment = True
       if( line.strip().find("*/") >= 0 ):
           in_multiline_comment = False
       if not in_multiline_comment:
           if( in_header or line.find( classname + '::' ) >= 0 and line.find("(") >= 0 and line.strip().find("//") != 0 ) and line.find( ";" ) < 0:
               in_header = True
               fnheader = line.replace( classname + '::', '' )
               fnheader = fnheader.replace( '{', '' )
               fnheader = fnheader.replace( ') :', ')' )
               if fnheader.find(")") >= 0:
                   in_header = False
               fnheader = fnheader.strip().replace( ')', ');' )
               fnheader = fnheader.strip().replace( ';const', 'const;' )
               fnheader = fnheader.strip().replace( '; const', ' const;' )
               cog.outl( fnheader );
       line = f.readline()
    f.close()
    cog.outl('')

#    debug.close()

def add_templated():
#    debug = open('debug.txt', 'a' )
#    debug.write( 'foo\n')
#    debug.write( 'infile [' + cog.inFile + ']\n' )

    infile = cog.inFile
    cppfile = infile.replace('.h','.cpp')
    splitinfile = infile.replace('\\','/').split('/')
    infilename = splitinfile[ len(splitinfile) - 1 ]
    classname = infilename.replace('.h','')
    # cog.outl( '// classname: ' + classname )
    # cog.outl( '// cppfile: ' + infilename.replace('.h','.cpp' ) )
    f = open( cppfile, 'r')
    in_multiline_comment = False
    in_header = False;
    line = f.readline()
    cog.outl( '// generated, using cog:' )
    while( line != '' ):
       # cog.outl(line)
       if( line.strip().find("/*") >= 0 ):
           in_multiline_comment = True
       if( line.strip().find("*/") >= 0 ):
           in_multiline_comment = False
       if not in_multiline_comment:
           if( in_header or line.find( classname + '<T>::' ) >= 0 and line.find("(") >= 0 and line.strip().find("//") != 0 ) and line.find( ";" ) < 0:
               in_header = True
               fnheader = line.replace('template< typename T >', '' ).replace( classname + '<T>::', '' )
               fnheader = fnheader.replace( '{', '' )
               fnheader = fnheader.replace( ') :', ')' )
               if fnheader.find(")") >= 0:
                   in_header = False
               fnheader = fnheader.strip().replace( ')', ');' )
               fnheader = fnheader.strip().replace( ';const', 'const;' )
               fnheader = fnheader.strip().replace( '; const', ' const;' )
               cog.outl( fnheader );
       line = f.readline()
    f.close()
    cog.outl('')


