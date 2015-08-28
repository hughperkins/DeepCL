"""
Copyright Hugh Perkins 2014, 2015 hughperkins at gmail

This Source Code Form is subject to the terms of the Mozilla Public License, 
v. 2.0. If a copy of the MPL was not distributed with this file, You can 
obtain one at http://mozilla.org/MPL/2.0/.

simply import this module in your header file like:

   // [[[cog
   // import cog_addheaders
   // cog_addheaders.add()
   // ]]]
   // [[[end]]]

... and run cog on the header file, to generate the header declarations

"""

import cog


def add(classname=''):
#    debug = open('debug.txt', 'a')
#    debug.write('foo\n')
#    debug.write('infile [' + cog.inFile + ']\n')

    infile = cog.inFile
    splitinfile = infile.replace('\\','/').split('/')
    infilename = splitinfile[ len(splitinfile) - 1 ]
    if classname == '':
        classname = infilename.replace('.h','')
    cppfile = infile.replace('.h','.cpp')
    # cog.outl('// classname: ' + classname)
    # cog.outl('// cppfile: ' + infilename.replace('.h','.cpp'))
    f = open(cppfile, 'r')
    in_multiline_comment = False
    in_header = False;
    line = f.readline()
    cog.outl('// generated, using cog:')
    while(line != ''):
       # cog.outl(line)
       if(line.strip().find("/*") >= 0):
           in_multiline_comment = True
       if(line.strip().find("*/") >= 0):
           in_multiline_comment = False
       if not in_multiline_comment:
           if(in_header or (line.find(' ' + classname + '::') >= 0 or line.find('\r' + classname + '::') >= 0 or line.find('\n' + classname + '::') >= 0 or line.find(classname + '::') == 0 or line.find('*' + classname + '::') >= 0 or line.find('&' + classname + '::') >= 0) and line.find("(") >= 0 and line.strip().find("//") != 0) and line.find(";") < 0:
               in_header = True
               fnheader = line.replace(classname + '::', '')
               fnheader = fnheader.replace('{', '')
               fnheader = fnheader.replace(') :', ')')
               if fnheader.find(")") >= 0:
                   in_header = False
               fnheader = fnheader.strip().replace(')', ');')
               fnheader = fnheader.strip().replace(';const', 'const;')
               fnheader = fnheader.strip().replace('; const', ' const;')
               cog.outl(fnheader);
       line = f.readline()
    f.close()
    cog.outl('')


def add_templated():
#    debug = open('debug.txt', 'a')
#    debug.write('foo\n')
#    debug.write('infile [' + cog.inFile + ']\n')

    infile = cog.inFile
    cppfile = infile.replace('.h','.cpp')
    splitinfile = infile.replace('\\','/').split('/')
    infilename = splitinfile[ len(splitinfile) - 1 ]
    classname = infilename.replace('.h','')
    # cog.outl('// classname: ' + classname)
    # cog.outl('// cppfile: ' + infilename.replace('.h','.cpp'))
    f = open(cppfile, 'r')
    in_multiline_comment = False
    in_header = False;
    line = f.readline()
    cog.outl('// generated, using cog:')
    while(line != ''):
       # cog.outl(line)
       if(line.strip().find("/*") >= 0):
           in_multiline_comment = True
       if(line.strip().find("*/") >= 0):
           in_multiline_comment = False
       if not in_multiline_comment:
           if(in_header or line.find(classname + '<T>::') >= 0 and line.find("(") >= 0 and line.strip().find("//") != 0) and line.find(";") < 0:
               in_header = True
               fnheader = line.replace('template< typename T >', '').replace(classname + '<T>::', '')
               fnheader = fnheader.replace('{', '')
               fnheader = fnheader.replace(') :', ')')
               if fnheader.find(")") >= 0:
                   in_header = False
               fnheader = fnheader.strip().replace(')', ');')
               fnheader = fnheader.strip().replace(';const', 'const;')
               fnheader = fnheader.strip().replace('; const', ' const;')
               cog.outl(fnheader);
       line = f.readline()
    f.close()
    cog.outl('')

# compared to `add`, this handles access declarations like PUBLIC, PRIVATE, PROTECTED
def addv2(classname = '', default_access='private'):
    infile = cog.inFile
    splitinfile = infile.replace('\\','/').split('/')
    infilename = splitinfile[ len(splitinfile) - 1 ]
    if classname == '':
        classname = infilename.replace('.h','')
    cppfile = infile.replace('.h','.cpp')
    f = open(cppfile, 'r')
    in_multiline_comment = False
    in_header = False;
    line = f.readline()
    decs_by_acc = {}
    decs_by_acc['public'] = []
    decs_by_acc['private'] = []
    decs_by_acc['protected'] = []
    thisdec = ''
    thisaccess = default_access
    while(line != ''):
        # cog.outl(line)
        if(line.strip().find("/*") >= 0):
            in_multiline_comment = True
        if(line.strip().find("*/") >= 0):
            in_multiline_comment = False
        if not in_multiline_comment:
            if(in_header or (line.find(' ' + classname + '::') >= 0 or line.find('\r' + classname + '::') >= 0 or line.find('\n' + classname + '::') >= 0 or line.find(classname + '::') == 0 or line.find('*' + classname + '::') >= 0 or line.find('&' + classname + '::') >= 0) and line.find("(") >= 0 and line.strip().find("//") != 0) and line.find(";") < 0:
                in_header = True
                fnheader = line.replace(classname + '::', '')
                fnheader = fnheader.replace('{', '')
                fnheader = fnheader.replace(') :', ')')
                if "PUBLIC " in fnheader:
                    thisaccess = 'public'
                if "PROTECTED " in fnheader:
                    thisaccess = 'protected'
                if "PRIVATE " in fnheader:
                    thisaccess = 'private'
                fnheader = fnheader.replace(thisaccess.upper() + ' ', '')
                got_all_header = False
                if fnheader.find(")") >= 0:
                    in_header = False
                    got_all_header = True
                fnheader = fnheader.strip().replace(')', ');')
                fnheader = fnheader.strip().replace(';const', 'const;')
                fnheader = fnheader.strip().replace('; const', ' const;')
                # cog.outl(fnheader);
                if thisdec != '' and not got_all_header:
                    thisdec += '    '
                thisdec += fnheader + '\n'
                if got_all_header:
                    decs_by_acc[thisaccess].append(thisdec)
                    thisdec = ''
                    thisaccess = default_access
        line = f.readline()
    f.close()

    cog.outl('// generated, using cog:')
    for accessor in ['public', 'protected', 'private']:
        decs = decs_by_acc[accessor]
        if len(decs) > 0:
            cog.outl('')
            cog.outl('{accessor}:'.format(accessor=accessor))
            for dec in decs:
                cog.out(dec)
    cog.outl('')

