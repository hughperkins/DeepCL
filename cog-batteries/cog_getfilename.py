# Copyright Hugh Perkins 2014,2015 hughperkins at gmail
#
# This Source Code Form is subject to the terms of the Mozilla Public License, 
# v. 2.0. If a copy of the MPL was not distributed with this file, You can 
# obtain one at http://mozilla.org/MPL/2.0/.

import cog
import os.path

def get_file_name():
    infile = cog.inFile
    cppfile = infile.replace('.h','.cpp')
    splitinfile = infile.replace('\\','/').split('/')
    infilename = splitinfile[ len(splitinfile) - 1 ]
    return classname

def get_class_name():
    infile = cog.inFile
    cppfile = infile.replace('.h','.cpp')
    splitinfile = infile.replace('\\','/').split('/')
    infilename = splitinfile[ len(splitinfile) - 1 ]
    ( classname, _ ) = os.path.splitext( infilename )
    return classname

